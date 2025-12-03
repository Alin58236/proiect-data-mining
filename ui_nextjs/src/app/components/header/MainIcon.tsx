'use client'

import React from 'react'
import Image from 'next/image'

const MainIcon = () => {
  return (
    <div className='flex items-center'>
      <Image
        src='/favicon.ico'
        alt='Main Icon'
        width={50}
        height={50}
        className='opacity-100'
        priority={false} // set true if you want to prioritize loading
      />
    </div>
  )
}

export default MainIcon
