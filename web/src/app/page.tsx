'use client';

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import dynamic from 'next/dynamic';
import { useState } from 'react';
import { Search } from 'lucide-react';

// Dynamic import of ForceGraph to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph').then(mod => mod.ForceGraph2D), {
  ssr: false,
  loading: () => <div className="w-full bg-gray-100 animate-pulse rounded-lg" />
});

// Sample data structure
const sampleData = {
  nodes: [
    {
      id: '1',
      name: 'Knowledge Graph Paper',
      type: 'paper',
      description: 'Research on knowledge graphs',
      paper: 'https://example.com/paper.pdf'
    },
    {
      id: '2',
      name: 'Graph Algorithm',
      type: 'code',
      codeRepo: 'https://github.com/example/repo',
      codePart: 'src/algorithm.py'
    }
  ],
  links: [
    { source: '1', target: '2', value: 1 }
  ]
};

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);
  const [graphData, setGraphData] = useState(sampleData);

  const handleSearch = () => {
    console.log('Searching:', searchQuery);
    // TODO: Implement search logic
  };

  const handleNodeClick = (node) => {
    setSelectedNode(node);
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
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
            <Search className="absolute left-3 top-2.5 text-gray-400" />
          </div>
          <Button onClick={handleSearch}>Search</Button>
        </div>

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
                  {selectedNode.type && (
                    <div>
                      <h3 className="font-semibold">Type</h3>
                      <p className="text-gray-600">{selectedNode.type}</p>
                    </div>
                  )}
                  {selectedNode.description && (
                    <div>
                      <h3 className="font-semibold">Description</h3>
                      <p className="text-gray-600">{selectedNode.description}</p>
                    </div>
                  )}
                  {/* Additional node details based on type */}
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
