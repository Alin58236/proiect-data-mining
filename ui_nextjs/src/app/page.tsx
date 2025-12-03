import Header from "./components/header/Header";
import Footer from "./components/Footer";
import Content from "./components/Content";


export default function Home() {
  return (
    
      <main className="flex flex-col h-screen w-screen items-center justify-between p-0 bg-black">
        <Header />
        <Content />
        <Footer />
      </main>
    
  );
}
