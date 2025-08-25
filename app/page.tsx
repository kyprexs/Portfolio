'use client'

import { useState } from 'react'

export default function Home() {
  const [currentSection, setCurrentSection] = useState('profile')

  const renderContent = () => {
    switch (currentSection) {
      case 'profile':
        return (
          <div className="space-y-6">
            {/* Personal Info Panels */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="data-panel">
                <div className="mb-4 font-bold">RESEARCHER ID #1337</div>
                <div className="space-y-2">
                  <div>&gt; NAME: Alex West</div>
                  <div>&gt; ROLE: ML Engineer & Researcher</div>
                  <div>&gt; YEARS: 4+ Experience</div>
                  <div>&gt; LOCATION: Victoria, BC</div>
                  <div>&gt; STATUS: Available for hire</div>
                </div>
              </div>
              
              <div className="data-panel">
                <div className="mb-4 font-bold">TECH STACK...</div>
                <div className="space-y-2">
                  <div>&gt; ML: TensorFlow, PyTorch, Scikit-learn</div>
                  <div>&gt; SCIENTIFIC: NumPy, Pandas, SciPy, Jupyter</div>
                  <div>&gt; LANGUAGES: Python, C++, Julia, R</div>
                  <div>&gt; TOOLS: Git, Docker, AWS, Linux</div>
                  <div>&gt; RESEARCH: Computer Vision, NLP, Optimization</div>
                </div>
              </div>
            </div>

            {/* Projects Section */}
            <div className="data-panel">
              <div className="mb-4 font-bold">PROJECTS PORTFOLIO</div>
              <div className="space-y-4">
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">
                    <a href="https://github.com/kyprexs/NeuralScript" target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-300 underline">
                      &gt; NEURALSCRIPT LANGUAGE
                    </a>
                  </div>
                  <div className="text-sm mb-2">Custom programming language for ML and scientific computing</div>
                  <div className="text-xs">TECH: Python, LLVM, SIMD, Memory Management</div>
                  <div className="text-xs mt-1">FEATURES: Unicode math operators, GPU acceleration, 10k+ LOC</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">
                    <a href="https://github.com/kyprexs/NeuralNetwork" target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-300 underline">
                      &gt; NEURAL NETWORK FRAMEWORK
                    </a>
                  </div>
                  <div className="text-sm mb-2">Neural network implementation built completely from scratch</div>
                  <div className="text-xs">TECH: Python, NumPy, Backpropagation, Optimization</div>
                  <div className="text-xs mt-1">FEATURES: Custom layers, training algorithms, model serialization</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">
                    <a href="https://github.com/kyprexs/Algok23" target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-300 underline">
                      &gt; ALGOK23 - ALGORITHMIC TRADING SYSTEM
                    </a>
                  </div>
                  <div className="text-sm mb-2">Advanced algorithmic trading platform with machine learning capabilities</div>
                  <div className="text-xs">TECH: Python, Machine Learning, PostgreSQL, Redis, Docker</div>
                  <div className="text-xs mt-1">FEATURES: Multi-exchange data, 150+ features, real-time ML inference</div>
                </div>
              </div>
            </div>
          </div>
        )
      
      case 'message':
        return (
          <div className="space-y-6">
            <div className="data-panel">
              <div className="mb-4 font-bold">ADDITIONAL PROJECTS</div>
              <div className="space-y-4">
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; COMPUTER VISION API</div>
                  <div className="text-sm mb-2">Real-time object detection and classification service</div>
                  <div className="text-xs">TECH: FastAPI, YOLO, Redis</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; ML MODEL DEPLOYMENT</div>
                  <div className="text-sm mb-2">Scalable model serving infrastructure</div>
                  <div className="text-xs">TECH: Kubernetes, TensorFlow Serving, Docker</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; DATA PIPELINE SYSTEM</div>
                  <div className="text-sm mb-2">Automated data processing and feature engineering</div>
                  <div className="text-xs">TECH: Apache Airflow, Spark, PostgreSQL</div>
                </div>
              </div>
            </div>
          </div>
        )
      
      case 'trophies':
        return (
          <div className="space-y-6">
            <div className="data-panel">
              <div className="mb-4 font-bold">RESEARCH & PUBLICATIONS</div>
              <div className="space-y-4">
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; CONFERENCE PAPERS</div>
                  <div className="text-sm mb-2">NeurIPS, ICML, CVPR - 3 publications</div>
                  <div className="text-xs">TOPICS: Computer Vision, Optimization</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; RESEARCH AREAS</div>
                  <div className="text-sm mb-2">Deep Learning, Scientific Computing, AI Ethics</div>
                  <div className="text-xs">SPECIALIZATION: ML for Scientific Applications</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; ACADEMIC BACKGROUND</div>
                  <div className="text-sm mb-2">MSc Computer Science, BSc Mathematics</div>
                  <div className="text-xs">FOCUS: Machine Learning & Statistics</div>
                </div>
              </div>
            </div>
          </div>
        )
      
      case 'inventory':
        return (
          <div className="space-y-6">
            <div className="data-panel">
              <div className="mb-4 font-bold">CONTACT INFORMATION</div>
              <div className="space-y-4">
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; EMAIL</div>
                  <div className="text-sm">x4xmails@gmail.com</div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; GITHUB</div>
                  <div className="text-sm">
                    <a href="https://github.com/kyprexs" target="_blank" rel="noopener noreferrer" className="text-white hover:text-gray-300 underline">
                      github.com/kyprexs
                    </a>
                  </div>
                </div>
                <div className="pixel-border p-3">
                  <div className="font-bold mb-2">&gt; AVAILABILITY</div>
                  <div className="text-sm">Open to full-time positions and consulting</div>
                  <div className="text-xs">REMOTE & ON-SITE: Victoria, BC</div>
                </div>
              </div>
            </div>
          </div>
        )
      
      default:
        return null
    }
  }

  return (
    <main className="min-h-screen bg-black text-white relative overflow-hidden">
      
      {/* Top Bar */}
      <div className="fixed top-4 right-4 z-50">
        <div className="flex items-center space-x-2">
          <a href="mailto:x4xmails@gmail.com" className="w-6 h-6 border border-white bg-black hover:bg-white hover:text-black transition-colors flex items-center justify-center text-xs">
            ✉
          </a>
          <a href="https://github.com/kyprexs" target="_blank" rel="noopener noreferrer" className="w-6 h-6 border border-white bg-black hover:bg-white hover:text-black transition-colors flex items-center justify-center text-xs">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
          </a>
          <a href="mailto:x4xmails@gmail.com?subject=Job Opportunity - Alex West" className="retro-button text-xs px-3 py-1 no-underline">
            HIRE ME
          </a>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 pt-16 pb-32">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column - Navigation */}
          <div className="lg:col-span-1 space-y-6">
            {/* Speech Bubble */}
            <div className="flex justify-center">
              <div className="pixel-border p-4 text-center">
                <span className="text-2xl">(•ᴗ•)و</span>
              </div>
            </div>

            {/* Achievement Badges */}
            <div className="space-y-4">
              <div className="pixel-border p-3">
                <div className="flex items-center mb-2">
                  <span className="mr-2">🏆</span>
                  <span className="font-bold">LANGUAGE CREATOR</span>
                </div>
                <div className="text-sm">NeuralScript achievement</div>
                <div className="text-xs mt-1 opacity-75">10k+ LOC • Production-ready</div>
              </div>
              
              <div className="pixel-border p-3">
                <div className="flex items-center mb-2">
                  <span className="mr-2">⚡</span>
                  <span className="font-bold">ML PIONEER</span>
                </div>
                <div className="text-sm">Custom neural network framework</div>
                <div className="text-xs mt-1 opacity-75">From scratch • Advanced features</div>
              </div>
            </div>

            {/* Navigation Menu */}
            <div className="nav-menu">
              <button 
                className={`nav-item w-full text-left border border-white p-3 mb-2 hover:bg-white hover:text-black transition-colors ${currentSection === 'profile' ? 'bg-blue-600 text-white font-bold' : 'bg-gray-600 text-white'}`}
                onClick={() => setCurrentSection('profile')}
              >
                ← HOME →
              </button>
              <button 
                className={`nav-item w-full text-left border border-white p-3 mb-2 hover:bg-white hover:text-black transition-colors ${currentSection === 'message' ? 'bg-blue-600 text-white font-bold' : 'bg-gray-600 text-white'}`}
                onClick={() => setCurrentSection('message')}
              >
                MORE PROJECTS
              </button>
              <button 
                className={`nav-item w-full text-left border border-white p-3 mb-2 hover:bg-white hover:text-black transition-colors ${currentSection === 'trophies' ? 'bg-blue-600 text-white font-bold' : 'bg-gray-600 text-white'}`}
                onClick={() => setCurrentSection('trophies')}
              >
                RESEARCH
              </button>
              <button 
                className={`nav-item w-full text-left border border-white p-3 mb-2 hover:bg-white hover:text-black transition-colors ${currentSection === 'inventory' ? 'bg-blue-600 text-white font-bold' : 'bg-gray-600 text-white'}`}
                onClick={() => setCurrentSection('inventory')}
              >
                CONTACT
              </button>
            </div>
          </div>

          {/* Right Column - Dynamic Content */}
          <div className="lg:col-span-2 space-y-6">
            {renderContent()}
          </div>
        </div>
      </div>



      {/* Footer */}
      <div className="fixed bottom-4 right-4">
        <span className="text-sm">v1.0.0</span>
      </div>
    </main>
  )
}
