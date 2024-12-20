'use client';

import { useState, useCallback } from 'react';
import { Network } from 'react-vis-network-graph';
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

// Convert data to vis-network format
const getVisData = (data: GraphData) => {
  const nodes = data.nodes.map(node => ({
    id: node.id,
    label: node.name,
    title: node.description,
    group: node.type
  }));

  const edges = data.links.map(link => ({
    from: link.source,
    to: link.target,
    value: link.value
  }));

  return { nodes, edges };
};

// Network options
const options = {
  nodes: {
    shape: 'dot',
    size: 16,
    font: {
      size: 14
    }
  },
  edges: {
    width: 1,
    smooth: {
      type: 'continuous'
    }
  },
  physics: {
    stabilization: false,
    barnesHut: {
      gravitationalConstant: -80000,
      springConstant: 0.001,
      springLength: 200
    }
  },
  groups: {
    paper: { color: '#97C2FC' },
    code: { color: '#FB7E81' },
    math: { color: '#7BE141' }
  }
};

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [graphData, setGraphData] = useState<GraphData>(sampleData);

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

  const handleNodeClick = useCallback((params: any) => {
    if (params.nodes && params.nodes[0]) {
      const nodeId = params.nodes[0];
      const node = sampleData.nodes.find(n => n.id === nodeId);
      if (node) {
        setSelectedNode(node);
      }
    }
  }, []);

  const visData = getVisData(graphData);

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
          <div className="w-full lg:w-2/3 h-[600px] border rounded-lg overflow-hidden bg-white">
            <Network
              data={visData}
              options={options}
              events={{
                click: handleNodeClick
              }}
              style={{ width: '100%', height: '100%' }}
            />
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
