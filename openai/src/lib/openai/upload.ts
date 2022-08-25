import { withError } from 'utils/error-handler';
import { openai } from './api';
import { ReadStream } from 'node:fs';

export const upoadFile = async (
  file: ReadStream,
  purpose: 'fine-tune' | 'search' | 'classifications' | 'answers'
) => {
  const { data, error } = await withError(() =>
    openai.createFile(file, purpose)
  );
  if (error) return { data: null, error };
  return { data: { fileId: data?.id }, error };
};
