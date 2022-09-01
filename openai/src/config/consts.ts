import { Consts, FineTuneConfig } from './openai.types';

export const fineTuneConfig: FineTuneConfig = {
  model: 'curie',
  batchSize: 4,
  iters: 4,
  modelName: 'shortener-v0-2',
};

const TIME_MARK = '2022-07-27-10-54-50';
const trained_model = `${
  fineTuneConfig.model
}:ft-lighthouse-feed:${fineTuneConfig.modelName.toLowerCase()}-${TIME_MARK}`;

export const consts: Consts = {
  fineTunnedModel: trained_model,
  maxPromptLength: 150,
  minTokens: 15,
  maxTokens: 22,
  maxTokenDownSample: 0.7,
  promptStart: 'Long: ',
  promptEnd: '',
  completionStart: 'Short: ',
  completionEnd: '.',
};
