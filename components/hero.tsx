"use client"

import { useEffect, useState } from "react"
import { ChevronDown, Github, MessageCircle, Mail } from "lucide-react"

export function Hero() {
  const [text, setText] = useState("")
  const fullText = "Backend Engineer & Automation Specialist"

  useEffect(() => {
    let index = 0
    const timer = setInterval(() => {
      setText(fullText.slice(0, index))
      index++
      if (index > fullText.length) {
        clearInterval(timer)
      }
    }, 100)

    return () => clearInterval(timer)
  }, [])

  return (
    <section id="home" className="min-h-screen flex items-center justify-center relative px-4">
      <div className="text-center max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-5xl md:text-7xl font-bold mb-4 tracking-tight">
            Hi, I'm{" "}
            <span className="text-[#00FFFF] relative">
              Alex
              <div className="absolute -bottom-2 left-0 w-full h-1 bg-gradient-to-r from-[#00FFFF] to-transparent"></div>
            </span>
          </h1>
          <div className="text-xl md:text-2xl text-[#B0B0B0] font-light h-8">
            {text}
            <span className="animate-pulse text-[#00FFFF]">|</span>
          </div>
        </div>

        <p className="text-lg md:text-xl text-[#B0B0B0] mb-12 max-w-2xl mx-auto leading-relaxed">
          I build robust backend systems, automate complex workflows, and create seamless user experiences. Passionate
          about turning ideas into scalable solutions.
        </p>

        <div className="flex justify-center space-x-6 mb-16">
          <a
            href="#contact"
            className="bg-[#00FFFF] text-[#0F0F0F] px-8 py-3 rounded-lg font-semibold hover:bg-[#00FFFF]/90 transition-all duration-200 transform hover:scale-105"
          >
            Get In Touch
          </a>
          <a
            href="#projects"
            className="border border-[#2E2E2E] px-8 py-3 rounded-lg font-semibold hover:border-[#00FFFF] hover:text-[#00FFFF] transition-all duration-200"
          >
            View Work
          </a>
        </div>
        {/* Social Icons Row */}
        <div className="flex justify-center space-x-8 mb-8">
          <a href="https://github.com/kyprexs" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-[#B0B0B0] hover:text-[#00FFFF] transition-colors duration-200 text-2xl">
            <Github size={32} />
          </a>
          <a href="https://discord.com/users/380541064848736256" target="_blank" rel="noopener noreferrer" aria-label="Discord" className="text-[#B0B0B0] hover:text-[#00FFFF] transition-colors duration-200 text-2xl">
            <MessageCircle size={32} />
          </a>
          <a href="mailto:x4xmails@gmail.com?subject=Project%20Possibility" aria-label="Email" className="text-[#B0B0B0] hover:text-[#00FFFF] transition-colors duration-200 text-2xl">
            <Mail size={32} />
          </a>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <ChevronDown className="text-[#00FFFF]" size={32} />
      </div>
    </section>
  )
}
