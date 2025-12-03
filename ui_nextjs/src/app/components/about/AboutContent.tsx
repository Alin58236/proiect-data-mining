import React from "react";
import Image from "next/image";

const AboutContent = () => {
  return (
    <div className="relative flex flex-col justify-center items-center w-full h-screen text-white">
      <Image
        src="/image4.jpg"
        alt="Background"
        layout="fill"
        objectFit="cover"
        quality={100}
        priority
        className="opacity-10"
      />
      <div className="relative z-10 p-6">
        <h1 className="text-4xl font-bold">Ideea proiectului</h1>
        <p>Am creat acest site pentru a putea demonstra si analiza intr-un mod interactiv diferentele dintre clasificarile DT, KNN, XGBoost, si MLP</p>
        <br />
        <h1 className="text-3lx font-bold">Cum functioneaza?</h1>
        <p>Utilizatorul selecteaza tipul de model sio parametri cu care vrea sa antreneze modelul. Frontend-ul face un API call catre backend, unde modelul este antrenat conform datelor de la user</p>
        <br />
        <h1 className="text-3lx font-bold">Rezultatul</h1>
        <p>Rezultatul vine sub forma unei matrici --- mai ziceti voi ce sa scriu aici ---</p>
      
      
      </div>
    </div>
  );
};

export default AboutContent;
