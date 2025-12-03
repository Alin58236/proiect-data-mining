export type ModelParam = {
  name: string;
  label: string;
  type: 'number' | 'text' | 'select';
  options?: (string | number)[];
  description: string;
};

export type ModelsRecord = Record<string, { params: ModelParam[] }>;

export type ParamsValues = Record<string, string | number>;