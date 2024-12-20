export interface Node {
  id: string;
  name: string;
  type: string;
  description?: string;
  codeRepo?: string;
  codePart?: string;
  paper?: string;
  mathExpression?: string;
  logicExpression?: string;
  image?: string;
}

export interface Link {
  source: string;
  target: string;
  value: number;
  description?: string;
}

export interface GraphData {
  nodes: Node[];
  links: Link[];
}
