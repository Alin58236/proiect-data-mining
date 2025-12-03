import React from 'react';
import Header from '../components/header/Header';
import Footer from '../components/Footer';
import AboutContent from '../components/about/AboutContent';

export default function AboutPage() {
  return (
    <>
    
    <main className="flex flex-col h-screen w-screen items-center justify-between p-0 bg-black overflow-hidden">
        <Header />
            <AboutContent />
        <Footer />
    </main>

    
    
    </>
    
  );
}
