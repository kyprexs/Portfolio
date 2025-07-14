import { ExternalLink, Github, Database, Zap, Globe, Bot } from "lucide-react"

export function Projects() {
  const projects = [
    {
      title: "Neural Network from Scratch",
      description:
        "Implemented a neural network from the ground up in Python, including forward and backward propagation, custom activation functions, and visualization tools. Great for learning the math and code behind deep learning.",
      tech: ["Python", "NumPy", "Matplotlib"],
      icon: <Zap className="text-[#00FFFF]" size={32} />, // You can change icon if you prefer
      github: "https://github.com/kyprexs/NeuralNetwork",
      demo: "https://github.com/kyprexs/NeuralNetwork",
    },
    {
      title: "Space Portfolio",
      description:
        "A visually stunning portfolio template inspired by space exploration. Features animated backgrounds, responsive layouts, and a modern tech stack. Perfect for developers who want a unique online presence.",
      tech: ["Next.js", "TailwindCSS", "Framer Motion"],
      icon: <Globe className="text-[#00FFFF]" size={32} />, // You can change icon if you prefer
      github: "https://github.com/kyprexs/spaceportfolio-template",
      demo: "https://github.com/kyprexs/spaceportfolio-template",
    },
    {
      title: "Medieval Portfolio",
      description:
        "A creative portfolio template with a medieval theme, including custom illustrations, parchment-style backgrounds, and interactive elements. Designed for creative professionals and history enthusiasts.",
      tech: ["React", "Styled Components", "SVG"],
      icon: <Database className="text-[#00FFFF]" size={32} />, // You can change icon if you prefer
      github: "https://github.com/kyprexs/Medevil-Portfolio",
      demo: "https://github.com/kyprexs/Medevil-Portfolio",
    },
  ]

  return (
    <section id="projects" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 uppercase tracking-wider">Featured Projects</h2>
          <div className="w-20 h-1 bg-[#00FFFF] mx-auto"></div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <div
              key={index}
              className="bg-[#1A1A1A] border border-[#2E2E2E] rounded-lg p-8 hover:border-[#00FFFF]/50 transition-all duration-300 group focus:outline-none focus:ring-2 focus:ring-[#00FFFF]"
              style={{ cursor: 'pointer' }}
            >
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="group-hover:scale-110 transition-transform duration-300">{project.icon}</div>
                  <h3 className="text-xl md:text-2xl font-bold text-white">{project.title}</h3>
                </div>
                <div className="flex space-x-3">
                  <a
                    href={project.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[#B0B0B0] hover:text-[#00FFFF] transition-colors duration-200"
                    aria-label={`View ${project.title} on GitHub`}
                  >
                    <Github size={20} />
                  </a>
                  <a
                    href={project.demo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[#B0B0B0] hover:text-[#00FFFF] transition-colors duration-200"
                    aria-label={`View ${project.title} demo`}
                  >
                    <ExternalLink size={20} />
                  </a>
                </div>
              </div>

              <p className="text-[#B0B0B0] text-lg leading-relaxed mb-6">{project.description}</p>

              <div className="flex flex-wrap gap-2">
                {project.tech.map((tech, techIndex) => (
                  <span
                    key={techIndex}
                    className="bg-[#2E2E2E] text-[#00FFFF] px-3 py-1 rounded-full text-sm font-medium"
                  >
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="text-center mt-12">
          <a
            href="https://github.com/kyprexs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-2 text-[#00FFFF] hover:text-white transition-colors duration-200 font-semibold"
          >
            <span>View All Projects</span>
            <ExternalLink size={16} />
          </a>
        </div>
      </div>
    </section>
  )
}
