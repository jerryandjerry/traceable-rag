import * as api from '@/api'
import { useRequest } from 'ahooks'
import { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import SpriteText from 'three-spritetext'
import styles from './index.module.scss'
import { renderGraphInfo } from './info'

interface GraphKeyDefinition {
  name: string | null
}

interface GraphNode {
  id: string
  entity_type: string
  description: string
  degree: number
  docnm: string
  color?: string
  __nodeVal?: number
}

interface GraphLink {
  source: string
  target: string
  description: string
  keywords: string
  weight: number
}

const TypedForceGraph3D = ForceGraph3D<GraphNode, GraphLink>

type ForceGraphInstance = NonNullable<
  NonNullable<Parameters<typeof TypedForceGraph3D>[0]['ref']>['current']
>

function endpointId(endpoint: unknown): string {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return String(endpoint.id ?? '')
  }
  return String(endpoint ?? '')
}

export function GraphViewer() {
  const graphRef = useRef<HTMLDivElement>(null)
  const infoRef = useRef<HTMLDivElement>(null)
  const forceGraphRef = useRef<ForceGraphInstance>()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const {
    data: graphmlData,
    loading: dataLoading,
    error: dataError,
  } = useRequest(
    async () => {
      if (!api.graphml) {
        throw new Error('GraphML API not available')
      }

      // Resolve the graph through the tenant-scoped list endpoint.
      const listed = await api.graphml.listGraphMLFiles({
        params: { t: Date.now() },
      })
      const filename = listed.data?.files?.[0]?.filename
      if (!filename) {
        return null
      }

      const response = await api.graphml.getGraphMLFile(filename, {
        params: { t: Date.now() },
      })
      return response.data
    },
    {
      cacheKey: 'graphml',
      cacheTime: 0,
    },
  )

  useEffect(() => {
    const container = graphRef.current
    if (!graphmlData || !container) {
      // The account has no graph yet -- stop, rather than spin forever.
      if (!dataLoading) setLoading(false)
      return
    }

    let disposed = false
    const timers = new Set<number>()
    const schedule = (callback: () => void, delay: number) => {
      if (disposed) return
      const timer = window.setTimeout(() => {
        timers.delete(timer)
        if (!disposed) callback()
      }, delay)
      timers.add(timer)
    }

    const loadAndRenderGraph = () => {
      try {
        setLoading(true)
        setError(null)

        const parseGraphML = (xmlText: string) => {
          const parser = new DOMParser()
          const xmlDoc = parser.parseFromString(xmlText, 'text/xml')

          const keys: Record<string, GraphKeyDefinition> = {}
          const keyElements = xmlDoc.querySelectorAll('key')
          keyElements.forEach((key) => {
            keys[key.getAttribute('id') ?? ''] = {
              name: key.getAttribute('attr.name'),
            }
          })

          const nodes: GraphNode[] = []
          const nodeElements = xmlDoc.querySelectorAll('node')
          nodeElements.forEach((node) => {
            const nodeData = {
              id: node.getAttribute('id') ?? '',
              entity_type: 'unknown',
              description: '',
              degree: 0,
              docnm: 'unknown',
            }

            const dataElements = node.querySelectorAll('data')
            dataElements.forEach((data) => {
              const keyId = data.getAttribute('key')
              const key = keys[keyId ?? '']
              if (key) {
                if (key.name === 'entity_type') {
                  nodeData.entity_type = data.textContent ?? ''
                } else if (key.name === 'description') {
                  nodeData.description = data.textContent ?? ''
                } else if (key.name === 'degree') {
                  nodeData.degree = parseInt(data.textContent ?? '') || 0
                } else if (key.name === 'docnm') {
                  nodeData.docnm = data.textContent ?? ''
                }
              }
            })

            nodes.push(nodeData)
          })

          const links: GraphLink[] = []
          const edgeElements = xmlDoc.querySelectorAll('edge')
          edgeElements.forEach((edge) => {
            const linkData = {
              source: edge.getAttribute('source') ?? '',
              target: edge.getAttribute('target') ?? '',
              description: '',
              keywords: '',
              weight: 1.0,
            }

            const dataElements = edge.querySelectorAll('data')
            dataElements.forEach((data) => {
              const keyId = data.getAttribute('key')
              const key = keys[keyId ?? '']
              if (key) {
                if (key.name === 'description') {
                  linkData.description = data.textContent ?? ''
                } else if (key.name === 'keywords') {
                  linkData.keywords = data.textContent ?? ''
                } else if (key.name === 'weight') {
                  linkData.weight = parseFloat(data.textContent ?? '') || 1.0
                }
              }
            })

            links.push(linkData)
          })

          return { nodes, links }
        }

        const { nodes, links } = parseGraphML(graphmlData)

        // Omit disconnected nodes and two-node islands while keeping tree leaves.
        const nodeConnections = new Map<string, number>()

        links.forEach((link) => {
          nodeConnections.set(
            link.source,
            (nodeConnections.get(link.source) || 0) + 1,
          )
          nodeConnections.set(
            link.target,
            (nodeConnections.get(link.target) || 0) + 1,
          )
        })

        const isolatedPairs = links.filter(
          (link) =>
            nodeConnections.get(link.source) === 1 &&
            nodeConnections.get(link.target) === 1,
        )

        const isolatedPairNodeIds = new Set<string>()
        isolatedPairs.forEach((pair) => {
          isolatedPairNodeIds.add(pair.source)
          isolatedPairNodeIds.add(pair.target)
        })

        const filteredNodes = nodes.filter((node) => {
          const connections = nodeConnections.get(node.id) || 0
          return connections === 0 ? false : !isolatedPairNodeIds.has(node.id)
        })

        const filteredNodeIds = new Set(filteredNodes.map((n) => n.id))

        const filteredLinks = links.filter(
          (link) =>
            filteredNodeIds.has(link.source) &&
            filteredNodeIds.has(link.target),
        )

        container.replaceChildren()
        const root = createRoot(container)
        const graphData = { nodes: filteredNodes, links: filteredLinks }
        let graphConfigured = false

        const configureGraph = () => {
          if (graphConfigured || !forceGraphRef.current) return

          const graph = forceGraphRef.current
          graph.d3Force('link')?.distance?.(50)
          graph.d3Force('link')?.strength?.(0.5)
          graph.d3Force('center')?.strength?.(1)
          graph.cameraPosition({ x: 0, y: 0, z: 100 })

          const camera = graph.camera()
          if (
            camera instanceof THREE.PerspectiveCamera ||
            camera instanceof THREE.OrthographicCamera
          ) {
            camera.zoom = 0.6
            camera.updateProjectionMatrix()
          }
          graphConfigured = true
        }

        const renderGraph = (width: number, height: number) => {
          if (disposed || width === 0 || height === 0) return

          root.render(
            <TypedForceGraph3D
              ref={forceGraphRef}
              graphData={graphData}
              nodeLabel="id"
              nodeAutoColorBy="docnm"
              nodeVal={(node) =>
                node.docnm && node.docnm.includes(' | ') ? 3 : 2
              }
              backgroundColor="#333333"
              linkColor={() => 'white'}
              width={width}
              height={height}
              enableNavigationControls
              cooldownTicks={100}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.4}
              nodeRelSize={4}
              onEngineTick={configureGraph}
              onEngineStop={() => {
                configureGraph()
                schedule(() => forceGraphRef.current?.zoomToFit(1000), 100)
              }}
              nodeThreeObjectExtend={false}
              nodeThreeObject={(node) => {
                const group = new THREE.Group()

                if (node.docnm && node.docnm.includes(' | ')) {
                  const geometry = new THREE.SphereGeometry(
                    node.__nodeVal || 2,
                    16,
                    12,
                  )
                  const material = new THREE.MeshLambertMaterial({
                    color: 0xffffff,
                  })
                  group.add(new THREE.Mesh(geometry, material))

                  const outlineGeometry = new THREE.SphereGeometry(
                    (node.__nodeVal || 2) + 0.3,
                    16,
                    12,
                  )
                  const outlineMaterial = new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.8,
                  })
                  group.add(new THREE.Mesh(outlineGeometry, outlineMaterial))
                } else {
                  const geometry = new THREE.SphereGeometry(
                    node.__nodeVal || 2,
                    16,
                    12,
                  )
                  const material = new THREE.MeshLambertMaterial({
                    color: node.color,
                  })
                  group.add(new THREE.Mesh(geometry, material))
                }

                const sprite = new SpriteText(String(node.id ?? ''))
                sprite.color =
                  node.docnm && node.docnm.includes(' | ')
                    ? '#ffffff'
                    : (node.color ?? '#ffffff')
                sprite.textHeight = 6
                sprite.center.y = -0.6
                group.add(sprite)

                return group
              }}
              linkThreeObjectExtend
              linkThreeObject={(link) => {
                const text =
                  link.keywords || link.description.substring(0, 20) + '...'
                const sprite = new SpriteText(text)
                sprite.color = 'white'
                sprite.textHeight = 2
                return sprite
              }}
              linkPositionUpdate={(sprite, { start, end }) => {
                sprite.position.set(
                  start.x + (end.x - start.x) / 2,
                  start.y + (end.y - start.y) / 2,
                  start.z + (end.z - start.z) / 2,
                )
              }}
              onNodeHover={(node) => {
                const infoPanel = infoRef.current
                if (node && infoPanel) {
                  const description = `${node.description.substring(0, 100)}${node.description.length > 100 ? '...' : ''}`
                  renderGraphInfo(infoPanel, `Node: ${node.id}`, [
                    ['Document', node.docnm],
                    ['Type', node.entity_type],
                    ['Degree', node.degree],
                    ['Description', description],
                  ])
                }
              }}
              onLinkHover={(link) => {
                const infoPanel = infoRef.current
                if (link && infoPanel) {
                  const description = `${link.description.substring(0, 100)}${link.description.length > 100 ? '...' : ''}`
                  renderGraphInfo(
                    infoPanel,
                    `Link: ${endpointId(link.source)} → ${endpointId(link.target)}`,
                    [
                      ['Weight', link.weight],
                      ['Keywords', link.keywords],
                      ['Description', description],
                    ],
                  )
                }
              }}
            />,
          )

          const infoPanel = infoRef.current
          if (infoPanel) {
            const documents = [
              ...new Set(
                filteredNodes
                  .flatMap((node) =>
                    node.docnm.split(' | ').map((name: string) => name.trim()),
                  )
                  .filter((name: string) => name && name !== 'unknown'),
              ),
            ].join(', ')
            renderGraphInfo(infoPanel, 'GraphML Viewer', [
              ['Nodes', filteredNodes.length],
              ['Links', filteredLinks.length],
              ['Documents', documents],
            ])
          }

          schedule(() => {
            const canvas = container.querySelector('canvas')
            if (canvas) {
              canvas.style.maxWidth = '100%'
              canvas.style.maxHeight = '100%'
            }
            forceGraphRef.current?.zoomToFit(1000)
            schedule(() => forceGraphRef.current?.zoomToFit(1000), 500)
          }, 100)
        }

        const renderAtContainerSize = (retryIfEmpty: boolean) => {
          const width = container.offsetWidth
          const height = container.offsetHeight
          if (width === 0 || height === 0) {
            if (retryIfEmpty) {
              schedule(() => renderAtContainerSize(false), 100)
            }
            return
          }
          renderGraph(width, height)
        }

        const handleResize = () => renderAtContainerSize(false)
        window.addEventListener('resize', handleResize)
        schedule(() => renderAtContainerSize(true), 50)

        return () => {
          disposed = true
          window.removeEventListener('resize', handleResize)
          timers.forEach((timer) => window.clearTimeout(timer))
          timers.clear()
          root.unmount()
          forceGraphRef.current = undefined
        }
      } catch (err) {
        console.error('Failed to load graph data')
        setError(
          err instanceof Error ? err.message : 'Failed to load graph data',
        )
      } finally {
        setLoading(false)
      }
    }

    return loadAndRenderGraph()
  }, [graphmlData, dataLoading])

  if (error || dataError) {
    return (
      <div className={styles['graph-viewer']}>
        <div className={styles['error']}>
          <h3>Error loading graph</h3>
          <p>{error || dataError?.message}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles['graph-viewer']}>
      {(dataLoading || loading) && (
        <div className={styles['loading']}>
          <div className={styles['spinner']}></div>
          <p>Loading graph...</p>
          {dataError && (
            <p style={{ color: 'red', fontSize: '12px' }}>
              Data Error: {(dataError as Error).message}
            </p>
          )}
        </div>
      )}
      {!dataLoading && !loading && !graphmlData && (
        <div className={styles['loading']}>
          <p>
            No graph yet. Upload a document in File Base and it will appear
            here.
          </p>
        </div>
      )}
      <div ref={graphRef} className={styles['graph-container']} />
      <div ref={infoRef} id="info" className={styles['info-panel']}>
        <div>
          <strong>GraphML Viewer</strong>
        </div>
        <div>Nodes: Loading...</div>
        <div>Links: Loading...</div>
        <div>Documents: Loading...</div>
      </div>
    </div>
  )
}
