"use client"

import { Github, MessageCircle, Mail } from "lucide-react"

export function Contact() {
  return (
    <section id="contact" className="py-20 px-4">
      <div className="max-w-4xl mx-auto text-center">
        <div className="mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 uppercase tracking-wider">Get In Touch</h2>
          <div className="w-20 h-1 bg-[#00FFFF] mx-auto mb-6"></div>
          <p className="text-[#B0B0B0] text-lg max-w-2xl mx-auto">
            Ready to discuss your next project? I'm always interested in new opportunities and challenging problems to
            solve.
          </p>
        </div>

        <div className="mb-12">
          <a
            href="mailto:x4xmails@gmail.com?subject=Project%20Possibility"
            className="inline-flex items-center bg-[#00FFFF] text-[#0F0F0F] px-8 py-4 rounded-lg font-semibold hover:bg-[#00FFFF]/90 transition-all duration-200 transform hover:scale-105 text-lg"
          >
            Contact Me
          </a>
        </div>
        {/* Social Icons Row */}
        <div className="flex justify-center space-x-8 mb-16">
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

        {/* Footer */}
        <div className="border-t border-[#2E2E2E] pt-8">
          <p className="text-[#B0B0B0]">© 2024 Alex Developer. Built with Next.js and Tailwind CSS.</p>
        </div>
      </div>
    </section>
  )
}
