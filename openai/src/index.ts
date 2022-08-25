import { createReadStream } from 'node:fs';
import { createFineTune, listFineTunes } from 'lib/openai/fine-tunning';
import { upoadFile } from 'lib/openai/upload';
import { consts, fineTuneConfig } from 'config/consts';
import { getCompletion } from 'lib/openai/completions';
import { deleteModel } from 'lib/openai/deletion';

async function makeFineTuneModel() {
  const uploadTrainFile = async () => {
    const file = createReadStream('data/data.jsonl');
    const { data: uploadFileData } = await upoadFile(file, 'fine-tune');
    return uploadFileData?.fileId;
  };

  const fileId = await uploadTrainFile();
  if (!fileId) return;

  const { data: fineTuneData } = await createFineTune(fileId, fineTuneConfig);
  if (!fineTuneData) return;

  console.log(`Started fine-tune model with id ${fineTuneData.id}`);
  return fineTuneData.id;
}

async function listFineTuneModels() {
  const { data } = await listFineTunes();
  console.log('Active fine-tune models: ', data);
}

async function getModelCompletion() {
  const userPrompt =
    'Colchón Viscoelástico con núcleo HR Elio Grafeno, 150x190x24';

  console.log(`Model: ${consts.fineTunnedModel}`);
  console.log(`Prompt: "${userPrompt}"`);

  const { data, error } = await getCompletion(
    consts.fineTunnedModel,
    userPrompt,
    true,
    { temperature: 0.2 }
  );

  if (error) {
    console.error(error.message);
    return;
  }
  console.log(
    `Summary: "${data?.choices
      ?.at(0)
      ?.text?.replace(consts.completionStart, '')}"`
  );
}

async function deleteAllModels({ exceptions }: { exceptions?: string[] } = {}) {
  const { data: models } = await listFineTunes();
  if (!models) return;
  models.forEach(async ({ fine_tuned_model: model }) => {
    if (model && !exceptions?.includes(model)) {
      const { error } = await deleteModel(model);
      if (!error) console.log(`Model ${model} deleted successfully.`);
    }
  });
}

// getModelCompletion();
// makeFineTuneModel();
listFineTuneModels();
// deleteAllModels({ exceptions: [consts.fineTunnedModel] });
