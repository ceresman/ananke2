'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Alert } from '@/components/ui/alert';
import { GraphData, Node } from '@/types/graph';

// Sample data structure
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
    }
  ],
  links: [
    { source: '1', target: '2', value: 1 },
    { source: '2', target: '3', value: 1 }
  ]
};

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [graphData, setGraphData] = useState<GraphData>(sampleData);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleSearch = useCallback(() => {
    if (!searchQuery.trim()) return;
    setIsLoading(true);
    setError(null);

    try {
      const filteredNodes = sampleData.nodes.filter(node =>
        node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.description?.toLowerCase().includes(searchQuery.toLowerCase())
      );

      if (filteredNodes.length === 0) {
        setError('No results found');
        setGraphData({ nodes: [], links: [] });
        setIsLoading(false);
        return;
      }

      const nodeIds = new Set(filteredNodes.map(node => node.id));
      const relatedLinks = sampleData.links.filter(link =>
        nodeIds.has(link.source as string) || nodeIds.has(link.target as string)
      );

      setGraphData({
        nodes: filteredNodes,
        links: relatedLinks
      });
    } catch (err) {
      setError('Error searching graph data');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery]);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = containerRef.current.clientWidth;
    canvas.height = 600;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate node positions
    const nodePositions = new Map();
    graphData.nodes.forEach((node, index) => {
      const angle = (index / graphData.nodes.length) * 2 * Math.PI;
      const radius = Math.min(canvas.width, canvas.height) / 4;
      const x = canvas.width / 2 + radius * Math.cos(angle);
      const y = canvas.height / 2 + radius * Math.sin(angle);
      nodePositions.set(node.id, { x, y });
    });

    // Draw edges
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    graphData.links.forEach(link => {
      const sourcePos = nodePositions.get(link.source);
      const targetPos = nodePositions.get(link.target);
      if (sourcePos && targetPos) {
        ctx.beginPath();
        ctx.moveTo(sourcePos.x, sourcePos.y);
        ctx.lineTo(targetPos.x, targetPos.y);
        ctx.stroke();
      }
    });

    // Draw nodes
    graphData.nodes.forEach(node => {
      const pos = nodePositions.get(node.id);
      if (pos) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 8, 0, 2 * Math.PI);
        switch (node.type) {
          case 'paper':
            ctx.fillStyle = '#97C2FC';
            break;
          case 'code':
            ctx.fillStyle = '#FB7E81';
            break;
          case 'math':
            ctx.fillStyle = '#7BE141';
            break;
          default:
            ctx.fillStyle = '#666';
        }
        ctx.fill();

        // Draw node labels
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(node.name, pos.x, pos.y + 20);
      }
    });

    // Handle click events
    const handleClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Check if click is on a node
      for (const [nodeId, pos] of nodePositions.entries()) {
        const dx = x - pos.x;
        const dy = y - pos.y;
        if (dx * dx + dy * dy < 64) { // 8 * 8 = radius squared
          const node = graphData.nodes.find(n => n.id === nodeId);
          if (node) {
            setSelectedNode(node);
          }
          break;
        }
      }
    };

    canvas.addEventListener('click', handleClick);
    return () => canvas.removeEventListener('click', handleClick);
  }, [graphData]);

  return (
    <div className="container mx-auto p-4 min-h-screen">
      <div className="flex flex-col gap-4">
        <div className="flex gap-2">
          <Input
            placeholder="Search knowledge graph..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-md"
          />
          <Button onClick={handleSearch} disabled={isLoading}>
            {isLoading ? 'Searching...' : 'Search'}
          </Button>
        </div>

        {error && (
          <Alert variant="destructive">
            <p>{error}</p>
          </Alert>
        )}

        <div className="flex flex-col lg:flex-row gap-4">
          <div ref={containerRef} className="w-full lg:w-2/3 h-[600px] border rounded-lg overflow-hidden bg-white">
            <canvas ref={canvasRef} />
          </div>

          <div className="w-full lg:w-1/3">
            {selectedNode && (
              <Card className="p-4">
                <h3 className="text-lg font-semibold mb-2">{selectedNode.name}</h3>
                <p className="text-sm text-gray-600 mb-4">{selectedNode.description}</p>
                {selectedNode.type === 'paper' && (
                  <a href={selectedNode.paper} className="text-blue-500 hover:underline">
                    View Paper
                  </a>
                )}
                {selectedNode.type === 'code' && (
                  <div>
                    <a href={selectedNode.codeRepo} className="text-blue-500 hover:underline block">
                      View Repository
                    </a>
                    <p className="text-sm text-gray-500 mt-2">File: {selectedNode.codePart}</p>
                  </div>
                )}
                {selectedNode.type === 'math' && (
                  <div className="bg-gray-50 p-2 rounded">
                    <code>{selectedNode.mathExpression}</code>
                  </div>
                )}
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
