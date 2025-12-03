'use client'

import React, { useEffect, useState } from 'react'
import MainIcon from './MainIcon'
import Link from 'next/link'

const Header = () => {
  const [showButtons, setShowButtons] = useState(false)

  useEffect(() => {
    const checkScreenSize = () => {
      setShowButtons(window.innerWidth >= 640)
    }

    checkScreenSize()

    window.addEventListener('resize', checkScreenSize)
    return () => window.removeEventListener('resize', checkScreenSize)
  }, [])

  return (
    <header className='flex top-0 left-0 bg-[#181a1c] items-center text-[#F8FaF9] w-screen p-6'>
      <Link href={'/'}>
      <div id='logoDiv' className='flex max-w-80 items-center gap-3'>
        <MainIcon />
        <h1 className='text-5xl font-bold select-none'>trAIner</h1>
      </div>
      </Link>

      
        <nav id='menuDiv' className='ml-auto flex  text-xl select-none items-center h-full'>
          { showButtons ? <Link href={'/'}><h2 className='cursor-pointer hover:bg-indigo-800 transition p-5 rounded-3xl'>Home</h2></Link> : <div></div>}
          <Link href={'/about'}><h2 className='cursor-pointer hover:bg-indigo-800 transition p-5 rounded-3xl'>About</h2></Link>
        </nav>
       
    </header>
  )
}

export default Header
