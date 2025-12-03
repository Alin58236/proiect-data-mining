'use client'
import React, { ChangeEvent } from 'react'
import { useState } from 'react';
import { ModelResult } from '../types/ModelResult';
import { ParamsValues } from '../types/ModelParam';
import Button from './Button';
import { models } from '../data/models';

const FormComponent = () => {

  const [textInput, setTextInput] = useState<string>('');

  const [selectedModel, setSelectedModel] = useState<string>('DT');
  const [params, setParams] = useState<ParamsValues>({});
  const [result, setResult] = useState<ModelResult>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleModelChange(e: ChangeEvent<HTMLSelectElement>) {
    setSelectedModel(e.target.value);
    setParams({});
    console.log("Model changed to " + e.target.value);
  }

  function handleParamChange(name: string, value: string) {
    const paramType = models[selectedModel].params.find(p => p.name === name)?.type;
    let val: string | number = value;
    if (paramType === 'number') val = Number(value);
    setParams(prev => ({ ...prev, [name]: val }));
    console.log(`Param ${name} changed to ${val}`);
  }

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    try {
      const requestBody = { model: selectedModel, parameters: params };
      const response = await fetch('/api/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      console.log("Request body:", requestBody);
      if (!response.ok) throw new Error(`Nu are cum sa mearga inca: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err.message || 'Eroare necunoscută');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function handleTryAgain() {
    setResult(null);
    setParams({});
    setSelectedModel('DT');
    setError(null);
  }

  return (
    <div id='fullComponentContainer' className=' custom-scrollbar overflow-scroll flex flex-col p-16 lg:w-1/2 max-w-[1/2] w-full h-full bg-[#24282b]  text-white  '>

      <h2 className="text-2xl font-bold mb-6">Alege model și parametri</h2>

       {!result && (
          <>
            

            <div id="plm stie" className="mb-6">
              <label htmlFor="model-select" className="block mb-2 font-medium">
                Model:
              </label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={handleModelChange}
                className="w-full border border-secondary rounded px-3 py-2 bg-primary text-textcolor"
              >
                {Object.keys(models).map(model => (
                  <option className='bg-indigo-800' key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-6">
              {models[selectedModel].params.length === 0 && (
                <p className="text-accent italic">Acest model nu are parametrii suplimentari.</p>
              )}

              {models[selectedModel].params.map(({ name, label, type, options, description }) => (
                <div key={name}>
                  <label htmlFor={name} className="block mb-1 font-medium">
                    {label}:
                  </label>
                  {type === 'select' ? (
                    <select
                      id={name}
                      value={params[name]?.toString() ?? ''}
                      onChange={e => handleParamChange(name, e.target.value)}
                      className="w-full border border-secondary rounded px-3 py-2 bg-accent text-white"
                    >
                      <option className='bg-indigo-800' value="" disabled>
                        Selectează
                      </option>
                      {options!.map(opt => (
                        <option  key={opt.toString()} value={opt.toString()} className="bg-indigo-800 text-white">
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={name}
                      type={type}
                      value={params[name]?.toString() ?? ''}
                      onChange={e => handleParamChange(name, e.target.value)}
                      placeholder={description}
                      className="w-full border border-accent rounded px-3 py-2 bg-secondary text-white"
                    />
                  )}
                  <p className="text-xs text-gray-400 mt-1">{description}</p>
                </div>
              ))}
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading}
              className="mt-8 bg-181a1c hover:bg-indigo-800 text-white border font-semibold px-6 py-2 rounded-3xl transition disabled:opacity-50"
            >
              {loading ? "Rulează..." : "Antrenează model"}
            </button>

            {error && <p className="mt-4 text-red-500 font-semibold">Eroare: {error}</p>}
          </>
        )}


{/* De mutat result in cealalta componenta react */}
        {result && (
          <div>
            <h2 className="text-2xl font-bold mb-4">Rezultate</h2>

            <div className="mb-4">
              <h3 className="font-semibold">Matrice de confuzie:</h3>
              <table className="table-auto border-collapse border border-gray-600 w-full text-center mb-4">
                <tbody>
                  {result.confusion_matrix.map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j} className="border border-gray-600 px-2 py-1">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>
                <strong>Acuratețe:</strong> {result.accuracy.toFixed(3)}
              </p>
              <div className="mt-3">
                <h4>Clasa benignă:</h4>
                <p>Precizie: {result.precision_benign.toFixed(3)}</p>
                <p>Recall: {result.recall_benign.toFixed(3)}</p>
                <p>F1 Score: {result.f1_benign.toFixed(3)}</p>
              </div>
              <div className="mt-3">
                <h4>Clasa malignă:</h4>
                <p>Precizie: {result.precision_malignant.toFixed(3)}</p>
                <p>Recall: {result.recall_malignant.toFixed(3)}</p>
                <p>F1 Score: {result.f1_malignant.toFixed(3)}</p>
              </div>
            </div>

            <button
              onClick={handleTryAgain}
              className="mt-6 bg-white hover:bg-white text-white font-semibold px-6 py-2 rounded transition"
            >
              Try Again
            </button>
          </div>
        )}
    </div>
  )
}

export default FormComponent