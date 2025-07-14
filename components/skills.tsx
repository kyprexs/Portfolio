"use client"

import { useState } from "react"

export function Skills() {
  const [activeCategory, setActiveCategory] = useState("backend")

  const skillCategories = {
    backend: {
      title: "Backend & Infrastructure",
      skills: [
        { name: "Node.js", level: 95 },
        { name: "Python", level: 90 },
        { name: "PostgreSQL", level: 88 },
        { name: "MongoDB", level: 85 },
        { name: "Redis", level: 82 },
        { name: "Docker", level: 90 },
        { name: "Kubernetes", level: 75 },
        { name: "AWS/GCP", level: 85 },
      ],
    },
    automation: {
      title: "Automation & DevOps",
      skills: [
        { name: "CI/CD Pipelines", level: 92 },
        { name: "GitHub Actions", level: 90 },
        { name: "Jenkins", level: 85 },
        { name: "Terraform", level: 80 },
        { name: "Ansible", level: 78 },
        { name: "Monitoring", level: 85 },
        { name: "Scripting", level: 95 },
        { name: "Workflow Design", level: 88 },
      ],
    },
    frontend: {
      title: "Frontend & UI",
      skills: [
        { name: "React", level: 85 },
        { name: "Next.js", level: 82 },
        { name: "TypeScript", level: 88 },
        { name: "Tailwind CSS", level: 90 },
        { name: "Vue.js", level: 75 },
        { name: "JavaScript", level: 92 },
        { name: "HTML/CSS", level: 88 },
        { name: "Responsive Design", level: 85 },
      ],
    },
  }

  return (
    <section id="skills" className="py-20 px-4 bg-[#1A1A1A]/30">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 uppercase tracking-wider">Technical Skills</h2>
          <div className="w-20 h-1 bg-[#00FFFF] mx-auto"></div>
        </div>

        <div className="flex justify-center mb-12">
          <div className="bg-[#1A1A1A] border border-[#2E2E2E] rounded-lg p-2 flex">
            {Object.entries(skillCategories).map(([key, category]) => (
              <button
                key={key}
                onClick={() => setActiveCategory(key)}
                className={`px-6 py-3 rounded-md transition-all duration-200 text-sm uppercase tracking-wider ${
                  activeCategory === key
                    ? "bg-[#00FFFF] text-[#0F0F0F] font-semibold"
                    : "text-[#B0B0B0] hover:text-white"
                }`}
              >
                {category.title}
              </button>
            ))}
          </div>
        </div>

        <div className="max-w-4xl mx-auto">
          <h3 className="text-2xl font-bold mb-8 text-center text-[#00FFFF]">
            {skillCategories[activeCategory as keyof typeof skillCategories].title}
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {skillCategories[activeCategory as keyof typeof skillCategories].skills.map((skill, index) => (
              <div key={index} className="bg-[#1A1A1A] border border-[#2E2E2E] p-6 rounded-lg">
                <div className="flex justify-between items-center mb-3">
                  <span className="font-semibold text-white">{skill.name}</span>
                  <span className="text-[#00FFFF] text-sm">{skill.level}%</span>
                </div>
                <div className="w-full bg-[#2E2E2E] rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-[#00FFFF] to-[#00FFFF]/70 h-2 rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${skill.level}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
