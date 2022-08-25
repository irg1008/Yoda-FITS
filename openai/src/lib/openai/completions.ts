import { openai } from './api';
import { withError, WithError } from 'utils/error-handler';
import { CreateCompletionRequest, CreateCompletionResponse } from 'openai';
import { consts, fineTuneConfig } from 'config/consts';
import { priceOfNTokens, priceOfString } from 'lib/prices/prices-per-string';
import { encode } from 'lib/gpt-3-encoder';

export const getCompletion = async (
  model: string,
  userPrompt: string,
  verbose?: boolean,
  options?: CreateCompletionRequest
): Promise<WithError<CreateCompletionResponse>> => {
  if (userPrompt.length > consts.maxPromptLength)
    return {
      data: null,
      error: {
        name: 'PromptTooLongError',
        message: `Prompt is too long (${userPrompt.length}). Max length is ${consts.maxPromptLength} chars.`,
      },
    };

  const maxOutTokens = getMaxTokensForPrompt(userPrompt, verbose);

  const response = await withError(() =>
    openai.createCompletion(model, {
      prompt: `${consts.promptStart}${userPrompt}${consts.promptEnd}`,
      echo: false,
      stop: consts.completionEnd,
      max_tokens: maxOutTokens,
      temperature: 0.2,
      top_p: 0.5,
      ...options,
    })
  );

  if (response.data && verbose) logFinalData(userPrompt, response.data);

  return response;
};

const getMaxTokensForPrompt = (userPrompt: string, log: boolean = true) => {
  const finalPrompt = userPrompt + consts.promptEnd;
  const { nTokens, encodedStrs } = encode(finalPrompt);
  const maxOutTokens = Math.max(
    Math.min(consts.maxTokens, Math.floor(nTokens * consts.maxTokenDownSample)),
    consts.minTokens
  );

  if (log) {
    const totalTokens = nTokens + maxOutTokens;
    console.log(
      `> Prompt Length: ${nTokens} tokens (${finalPrompt.length} chars). Max out: ${maxOutTokens} tokens. Total: ${totalTokens} tokens.`
    );
    // console.log('> Tokens:', encodedStrs);
    console.log(
      `> Max Price: ${priceOfNTokens(
        totalTokens,
        fineTuneConfig.model
      ).usagePrice.toFixed(8)}$.`
    );
  }

  return maxOutTokens;
};

const logFinalData = (userPrompt: string, data: CreateCompletionResponse) => {
  if (data?.choices?.length) {
    const { text } = data.choices[0];
    const finalPrompt = userPrompt + consts.promptEnd;

    if (text) {
      const { usagePrice, encodedLength } = priceOfString(
        text,
        fineTuneConfig.model
      );
      const { usagePrice: promptPrice, encodedLength: promptNTokens } =
        priceOfString(finalPrompt, fineTuneConfig.model);

      console.log(`> Final Length: ${promptNTokens + encodedLength} tokens.`);
      console.log(`> Final Price: ${(usagePrice + promptPrice).toFixed(8)}$.`);
    }
  }
};
