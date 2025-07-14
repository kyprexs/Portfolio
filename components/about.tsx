import { Code, Database, Zap, Users } from "lucide-react"

export function About() {
  const highlights = [
    {
      icon: <Database className="text-[#00FFFF]" size={24} />,
      title: "Backend Architecture",
      description: "Designing scalable systems with microservices, APIs, and cloud infrastructure",
    },
    {
      icon: <Zap className="text-[#00FFFF]" size={24} />,
      title: "Automation Expert",
      description: "Building CI/CD pipelines, workflow automation, and deployment strategies",
    },
    {
      icon: <Code className="text-[#00FFFF]" size={24} />,
      title: "Full-Stack Capable",
      description: "Frontend development skills to bridge the gap between design and functionality",
    },
    {
      icon: <Users className="text-[#00FFFF]" size={24} />,
      title: "Team Collaboration",
      description: "Working effectively with cross-functional teams to deliver quality solutions",
    },
  ]

  return (
    <section id="about" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 uppercase tracking-wider">About Me</h2>
          <div className="w-20 h-1 bg-[#00FFFF] mx-auto"></div>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <h3 className="text-2xl md:text-3xl font-bold mb-6 text-[#00FFFF]">Crafting Digital Solutions</h3>
            <p className="text-[#B0B0B0] text-lg leading-relaxed mb-6">
              With a passion for backend engineering and automation, I specialize in building robust, scalable systems
              that power modern applications. My expertise spans from database design and API development to DevOps and
              workflow automation.
            </p>
            <p className="text-[#B0B0B0] text-lg leading-relaxed mb-8">
              I believe in writing clean, maintainable code and implementing best practices that ensure long-term
              success. Whether it's optimizing database queries, setting up CI/CD pipelines, or creating seamless user
              interfaces, I approach every project with attention to detail and a focus on performance.
            </p>
            <div className="flex flex-wrap gap-4">
              <span className="bg-[#1A1A1A] border border-[#2E2E2E] px-4 py-2 rounded-lg text-sm">
                5+ Years Experience
              </span>
              <span className="bg-[#1A1A1A] border border-[#2E2E2E] px-4 py-2 rounded-lg text-sm">
                50+ Projects Delivered
              </span>
              <span className="bg-[#1A1A1A] border border-[#2E2E2E] px-4 py-2 rounded-lg text-sm">Remote-First</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {highlights.map((item, index) => (
              <div
                key={index}
                className="bg-[#1A1A1A] border border-[#2E2E2E] p-6 rounded-lg hover:border-[#00FFFF]/50 transition-all duration-300 group"
              >
                <div className="mb-4 group-hover:scale-110 transition-transform duration-300">{item.icon}</div>
                <h4 className="text-lg font-semibold mb-2 text-white">{item.title}</h4>
                <p className="text-[#B0B0B0] text-sm leading-relaxed">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
