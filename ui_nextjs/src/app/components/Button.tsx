import React from 'react'

interface ButtonProps {
    disabled?: boolean;
    className?: string;
    onClick?: () => void;
    title: string;
}   


const Button = ({ disabled,className, onClick, title }: ButtonProps) => {
  return (
    <button className={className} disabled={disabled} onClick={onClick}>{title}</button>
  )
}

export default Button