'use client'
import React, { useState, useEffect } from 'react';
import Image from 'next/image';

const images = ['/image1.jpg', '/image2.jpg', '/image3.jpg', '/image4.jpg', '/image5.jpg'];

const Art = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  useEffect(() => {
    const randomIndex = Math.floor(Math.random() * images.length);
    setSelectedImage(images[randomIndex]);
  }, []);

  // Render nothing or a placeholder during SSR (before hydration)
  if (!selectedImage) {
    return (
      <div className="flex flex-col lg:w-3/4  w-full md:h-full h-0 bg-[#24282a] items-center justify-center text-white  ">
        <p>Loading image...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col lg:w-3/4  w-full lg:h-full h-0 bg-[#24282a]  text-white  ">

      <div className="relative w-full h-full overflow-hidden">
        <Image
          src={selectedImage}
          alt="Random Artwork"
          layout="fill"
          objectFit="cover"
          className="opacity-70"
          loading='eager'
          priority={false} // optionally set true if preferred
        />
      </div>
    </div>
  );
};

export default Art;
