import { readFileSync } from 'fs'
import { join } from 'path'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import matter from 'gray-matter'

// Required for static export - tells Next.js which slugs to pre-render
export async function generateStaticParams() {
  return [
    { slug: 'AgloK23_Executive_Summary' },
    { slug: 'AgloK23_Machine_Learning_Architecture' },
    { slug: 'AgloK23_Alternative_Data_AI_Trading' },
    { slug: 'AgloK23_Risk_Management_Analytics' },
  ]
}

export default function ResearchPaper({ params }: { params: { slug: string } }) {
  const { slug } = params
  
  // Read the markdown file directly from the filesystem
  let content = ''
  let metadata: any = {}
  
  try {
    const filePath = join(process.cwd(), 'public', 'research_papers', `${slug}.md`)
    const fileContent = readFileSync(filePath, 'utf8')
    const { data, content: markdownContent } = matter(fileContent)
    metadata = data
    content = markdownContent
  } catch (error) {
    content = '# Paper Not Found\n\nThe requested research paper could not be found.'
  }

  return (
    <main className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Top Bar */}
      <div className="fixed top-4 right-4 z-50">
        <div className="flex items-center space-x-2">
          <a href="/" className="w-6 h-6 border border-white bg-black hover:bg-white hover:text-black transition-colors flex items-center justify-center text-xs">
            ←
          </a>
          <a href="mailto:x4xmails@gmail.com" className="w-6 h-6 border border-white bg-black hover:bg-white hover:text-black transition-colors flex items-center justify-center text-xs">
            ✉
          </a>
          <a href="https://github.com/kyprexs" target="_blank" rel="noopener noreferrer" className="w-6 h-6 border border-white bg-black hover:bg-white hover:text-black transition-colors flex items-center justify-center text-xs">
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
          </a>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 pt-16 pb-32">
        <div className="max-w-4xl mx-auto">
          {/* Paper Header */}
          <div className="data-panel mb-8">
            <div className="mb-6">
              <a href="/" className="text-sm text-gray-400 hover:text-white transition-colors">
                ← Back to Portfolio
              </a>
            </div>
            
            {metadata.author && (
              <div className="text-sm text-gray-400 mb-2">
                <strong>Author:</strong> {metadata.author}
              </div>
            )}
            {metadata.date && (
              <div className="text-sm text-gray-400 mb-2">
                <strong>Date:</strong> {metadata.date}
              </div>
            )}
            {metadata.version && (
              <div className="text-sm text-gray-400 mb-4">
                <strong>Version:</strong> {metadata.version}
              </div>
            )}
          </div>

          {/* Paper Content */}
          <div className="data-panel">
            <div className="prose prose-invert prose-white max-w-none">
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ children }) => (
                    <h1 className="text-3xl font-bold mb-6 text-white border-b border-gray-700 pb-2">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-2xl font-bold mb-4 text-white mt-8">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-xl font-bold mb-3 text-white mt-6">
                      {children}
                    </h3>
                  ),
                  h4: ({ children }) => (
                    <h4 className="text-lg font-bold mb-2 text-white mt-4">
                      {children}
                    </h4>
                  ),
                  p: ({ children }) => (
                    <p className="mb-4 text-gray-300 leading-relaxed">
                      {children}
                    </p>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc list-inside mb-4 text-gray-300 space-y-1">
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal list-inside mb-4 text-gray-300 space-y-1">
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => (
                    <li className="text-gray-300">
                      {children}
                    </li>
                  ),
                  strong: ({ children }) => (
                    <strong className="font-bold text-white">
                      {children}
                    </strong>
                  ),
                  em: ({ children }) => (
                    <em className="italic text-gray-300">
                      {children}
                    </em>
                  ),
                  code: ({ children, className }) => {
                    const isInline = !className
                    if (isInline) {
                      return (
                        <code className="bg-gray-800 text-green-400 px-1 py-0.5 rounded text-sm">
                          {children}
                        </code>
                      )
                    }
                    return (
                      <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto mb-4">
                        <code className="text-green-400 text-sm">
                          {children}
                        </code>
                      </pre>
                    )
                  },
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-blue-500 pl-4 italic text-gray-300 mb-4">
                      {children}
                    </blockquote>
                  ),
                  table: ({ children }) => (
                    <div className="overflow-x-auto mb-4">
                      <table className="min-w-full border border-gray-700">
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="border border-gray-700 px-4 py-2 bg-gray-800 text-white font-bold">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-gray-700 px-4 py-2 text-gray-300">
                      {children}
                    </td>
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
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
