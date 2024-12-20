'use client';

import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Alert, AlertDescription } from "../components/ui/alert";
import dynamic from 'next/dynamic';
import { useState } from 'react';
import { Search } from 'lucide-react';
import type { Node, GraphData } from '../types/graph';

// Dynamic import of ForceGraph to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph').then(mod => mod.ForceGraph2D), {
  ssr: false,
  loading: () => <div className="w-full h-full bg-gray-100 animate-pulse rounded-lg" />
});

// Sample data structure demonstrating all node types
const sampleData: GraphData = {
  nodes: [
    {
      id: '1',
      name: 'Knowledge Graph Research',
      type: 'paper',
      description: 'Research on knowledge graph applications',
      paper: 'https://example.com/paper.pdf'
    },
    {
      id: '2',
      name: 'Graph Algorithm Implementation',
      type: 'code',
      description: 'Core graph processing algorithm',
      codeRepo: 'https://github.com/example/repo',
      codePart: 'src/algorithm.py'
    },
    {
      id: '3',
      name: 'Graph Theory Formula',
      type: 'math',
      description: 'Mathematical foundation of graph theory',
      mathExpression: 'G = (V, E) where V is vertices and E is edges'
    },
    {
      id: '4',
      name: 'Knowledge Inference Rule',
      type: 'logic',
      description: 'Logical inference rules for knowledge graphs',
      logicExpression: 'IF node(x) AND edge(x,y) THEN connected(x,y)'
    },
    {
      id: '5',
      name: 'Architecture Diagram',
      type: 'image',
      description: 'System architecture visualization',
      image: 'https://example.com/diagram.png'
    }
  ],
  links: [
    { source: '1', target: '2', value: 1, description: 'Implements algorithms from' },
    { source: '2', target: '3', value: 1, description: 'Based on formula' },
    { source: '3', target: '4', value: 1, description: 'Supports inference rules' },
    { source: '1', target: '5', value: 1, description: 'Visualized in' }
  ]
};

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [graphData, setGraphData] = useState<GraphData>(sampleData);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setError(null);
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));

      const filteredNodes = sampleData.nodes.filter(node =>
        node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.description?.toLowerCase().includes(searchQuery.toLowerCase())
      );

      if (filteredNodes.length === 0) {
        setError('No results found for your search query');
        return;
      }

      const nodeIds = new Set(filteredNodes.map(n => n.id));
      const relatedLinks = sampleData.links.filter(link =>
        nodeIds.has(link.source as string) || nodeIds.has(link.target as string)
      );

      relatedLinks.forEach(link => {
        const sourceId = link.source as string;
        const targetId = link.target as string;
        if (!nodeIds.has(sourceId)) {
          nodeIds.add(sourceId);
          filteredNodes.push(sampleData.nodes.find(n => n.id === sourceId)!);
        }
        if (!nodeIds.has(targetId)) {
          nodeIds.add(targetId);
          filteredNodes.push(sampleData.nodes.find(n => n.id === targetId)!);
        }
      });

      setGraphData({
        nodes: filteredNodes,
        links: relatedLinks
      });
    } catch (error) {
      setError('An error occurred while searching. Please try again.');
      console.error('Search failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = (node: any, event: MouseEvent) => {
    const typedNode = node as Node;
    setSelectedNode(typedNode);
    if (!expandedNodes.has(typedNode.id)) {
      const newExpandedNodes = new Set(expandedNodes);
      newExpandedNodes.add(typedNode.id);
      setExpandedNodes(newExpandedNodes);

      const connectedLinks = sampleData.links.filter(link =>
        link.source === typedNode.id || link.target === typedNode.id
      );

      const newNodes = new Set(graphData.nodes.map(n => n.id));
      const nodesToAdd: Node[] = [];

      connectedLinks.forEach(link => {
        const otherId = link.source === typedNode.id ? link.target : link.source;
        if (!newNodes.has(otherId as string)) {
          const otherNode = sampleData.nodes.find(n => n.id === otherId);
          if (otherNode) nodesToAdd.push(otherNode);
        }
      });

      if (nodesToAdd.length > 0 || connectedLinks.length > 0) {
        setGraphData(prev => ({
          nodes: [...prev.nodes, ...nodesToAdd],
          links: [...prev.links, ...connectedLinks.filter(link =>
            !prev.links.some(l =>
              l.source === link.source && l.target === link.target
            )
          )]
        }));
      }
    }
  };

  return (
    <main className="min-h-screen p-4 lg:p-8 bg-gray-50">
      <div className="max-w-screen-2xl mx-auto space-y-6">
        {/* Search Bar */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Input
              placeholder="Search knowledge graph..."
              value={searchQuery}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchQuery(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent) => {
                if (e.key === 'Enter') handleSearch();
              }}
              className="pl-10"
              disabled={isLoading}
            />
            <Search className="absolute left-3 top-2.5 text-gray-400" size={20} />
          </div>
          <Button onClick={handleSearch} disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Search'}
          </Button>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Graph Visualization */}
          <div className="lg:col-span-2">
            <Card className="p-4">
              <div className="aspect-square">
                <ForceGraph2D
                  graphData={graphData}
                  nodeLabel="name"
                  nodeAutoColorBy="type"
                  linkWidth={1}
                  onNodeClick={handleNodeClick}
                  linkLabel={(link) => (link as any).description}
                  nodeCanvasObject={(node, ctx, globalScale) => {
                    const label = (node as any).name;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Sans-Serif`;
                    ctx.fillStyle = 'rgba(255,255,255,0.8)';
                    ctx.fillText(label, (node as any).x + 6, (node as any).y + 4);
                  }}
                />
              </div>
            </Card>
          </div>

          {/* Node Details */}
          <div className="lg:col-span-1">
            <Card className="p-4">
              {selectedNode ? (
                <div className="space-y-4">
                  <h2 className="text-2xl font-bold">{selectedNode.name}</h2>
                  <div className="space-y-2">
                    <h3 className="font-semibold">Type</h3>
                    <p className="text-gray-600 capitalize">{selectedNode.type}</p>
                  </div>
                  {selectedNode.description && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Description</h3>
                      <p className="text-gray-600">{selectedNode.description}</p>
                    </div>
                  )}
                  {selectedNode.codeRepo && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Code Repository</h3>
                      <a href={selectedNode.codeRepo} target="_blank" rel="noopener noreferrer"
                         className="text-blue-600 hover:underline">{selectedNode.codeRepo}</a>
                      {selectedNode.codePart && (
                        <p className="text-gray-600">File: {selectedNode.codePart}</p>
                      )}
                    </div>
                  )}
                  {selectedNode.paper && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Academic Paper</h3>
                      <a href={selectedNode.paper} target="_blank" rel="noopener noreferrer"
                         className="text-blue-600 hover:underline">View Paper</a>
                    </div>
                  )}
                  {selectedNode.mathExpression && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Mathematical Expression</h3>
                      <p className="text-gray-600 font-mono bg-gray-50 p-2 rounded">
                        {selectedNode.mathExpression}
                      </p>
                    </div>
                  )}
                  {selectedNode.logicExpression && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Logical Expression</h3>
                      <p className="text-gray-600 font-mono bg-gray-50 p-2 rounded">
                        {selectedNode.logicExpression}
                      </p>
                    </div>
                  )}
                  {selectedNode.image && (
                    <div className="space-y-2">
                      <h3 className="font-semibold">Image</h3>
                      <img src={selectedNode.image} alt={selectedNode.name}
                           className="w-full rounded-lg shadow-sm" />
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-gray-500">Select a node to view details</p>
              )}
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
