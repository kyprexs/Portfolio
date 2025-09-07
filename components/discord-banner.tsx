'use client'

import { useState } from 'react'

export function DiscordBanner() {
  const [isVisible, setIsVisible] = useState(true)

  if (!isVisible) return null

  return (
    <div className="fixed top-0 left-0 right-0 z-[60] bg-black border-b-2 border-white">
      <div className="container mx-auto px-4 py-3">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center space-x-3">
            <div className="text-2xl">
              💬
            </div>
            <div className="text-white font-bold text-xs sm:text-sm md:text-base">
              <span className="hidden sm:inline">Got project ideas? Message me on Discord for FREE development!</span>
              <span className="sm:hidden">Got ideas? Message me on Discord!</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
