import React from 'react'
import FormComponent from './FormComponent'
import ArtComponent from './Art'

const Content = () => {
  return (
    <div className=' h-[80vh] flex w-screen flex-col lg:flex-row bg-[#24282b] items-center justify-center text-white '>
      
        <FormComponent />

        <ArtComponent />

      
    </div>
  )
}

export default Content