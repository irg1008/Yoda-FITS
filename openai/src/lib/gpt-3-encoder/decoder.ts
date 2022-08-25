import { decode as gpt3Decode } from 'gpt-3-encoder';

export const decode = (tokens: number[]): string => {
  return gpt3Decode(tokens);
};
