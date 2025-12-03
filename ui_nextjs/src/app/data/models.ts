import type { ModelsRecord } from '../types/ModelParam';

export const models: ModelsRecord = {
  DT: { params: [] },
  KNN: {
    params: [
      { name: 'n_neighbors', label: 'Număr vecini', type: 'number', description: 'Numărul de vecini folosiți la vot' },
      { name: 'weights', label: 'Weights', type: 'select', options: ['uniform', 'distance'], description: 'Cum votează vecinii' },
      { name: 'p', label: 'P', type: 'select', options: [1, 2], description: 'Modul de calcul al distanței: 1=Manhattan, 2=Euclidiană' },
    ],
  },
  MLP: {
    params: [
      { name: 'hidden_layer_sizes', label: 'Hidden layers', type: 'text', description: 'Numărul straturi ascunse și neuroni (ex: 100 sau 50,30)' },
      { name: 'activation', label: 'Activation', type: 'select', options: ['relu', 'tanh', 'logistic', 'identity'], description: 'Funcția de activare' },
      { name: 'solver', label: 'Solver', type: 'select', options: ['adam', 'sgd', 'lbfgs'], description: 'Algoritm optimizare (backpropagation)' },
      { name: 'max_iter', label: 'Max iter', type: 'number', description: 'Numărul maxim de epoci' },
    ],
  },
  XGBoost: {
    params: [
      { name: 'n_estimators', label: 'Nr. arbori', type: 'number', description: 'Numărul de arbori' },
      { name: 'scale_pos_weight', label: 'Scale pos weight', type: 'number', description: 'Raportul clasei majoritare/minoritate' },
      { name: 'eval_metric', label: 'Eval metric', type: 'text', description: 'Funcția de evaluare folosită' },
      { name: 'n_jobs', label: 'Nr. thread-uri', type: 'number', description: '-1 = toate core-urile' },
    ],
  },
};