
import React, { useRef, useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import { NodeData, Edge, CampaignStage } from '../types';
import Node from './Node';
import { Microscope } from 'lucide-react';

export interface CanvasHandle {
  flyTo: (x: number, y: number, zoom?: number) => void;
}

interface CanvasProps {
  nodes: NodeData[];
  edges: Edge[];
  onNodeAction: (action: string, nodeId: string) => void;
  selectedNodeId: string | null;
  onSelectNode: (id: string | null) => void;
  onNodeMove: (id: string, x: number, y: number) => void; // New prop for moving nodes
}

const Canvas = forwardRef<CanvasHandle, CanvasProps>(({ nodes, edges, onNodeAction, selectedNodeId, onSelectNode, onNodeMove }, ref) => {
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(0.8);
  
  // Canvas Panning State
  const [isPanning, setIsPanning] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  // Node Dragging State
  const [draggedNode, setDraggedNode] = useState<{ id: string; startX: number; startY: number; initialNodeX: number; initialNodeY: number } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>(0);

  const NODE_WIDTH = 320;

  // Expose flyTo method to parent
  useImperativeHandle(ref, () => ({
    flyTo: (targetX: number, targetY: number, targetZoom: number = 0.8) => {
       if (!containerRef.current) return;
       
       const { width, height } = containerRef.current.getBoundingClientRect();
       const endOffsetX = (width / 2) - (targetX * targetZoom);
       const endOffsetY = (height / 2) - (targetY * targetZoom);
       const startOffsetX = offset.x;
       const startOffsetY = offset.y;
       const startZoom = zoom;
       const startTime = performance.now();
       const duration = 1000;
       
       const animate = (currentTime: number) => {
         const elapsed = currentTime - startTime;
         const progress = Math.min(elapsed / duration, 1);
         const ease = 1 - Math.pow(1 - progress, 3);
         
         setOffset({
             x: startOffsetX + (endOffsetX - startOffsetX) * ease,
             y: startOffsetY + (endOffsetY - startOffsetY) * ease
         });
         setZoom(startZoom + (targetZoom - startZoom) * ease);
         
         if (progress < 1) {
           animationRef.current = requestAnimationFrame(animate);
         }
       };
       cancelAnimationFrame(animationRef.current || 0);
       animationRef.current = requestAnimationFrame(animate);
    }
  }));

  useEffect(() => {
    if (containerRef.current) {
      const { height } = containerRef.current.getBoundingClientRect();
      setOffset({ x: 100, y: height / 2 - 200 });
    }
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;

    // 1. Check if clicking a button/interactive element inside a node (Don't drag)
    if (target.closest('button') || target.closest('input') || target.closest('select')) return;

    // 2. Check if clicking a Node (Start Node Drag)
    const nodeElement = target.closest('.node-interactive');
    if (nodeElement) {
       const nodeId = nodeElement.getAttribute('data-id');
       if (nodeId) {
           const node = nodes.find(n => n.id === nodeId);
           if (node) {
               setDraggedNode({
                   id: nodeId,
                   startX: e.clientX,
                   startY: e.clientY,
                   initialNodeX: node.x,
                   initialNodeY: node.y
               });
               onSelectNode(nodeId); 
           }
       }
       return;
    }

    // 3. Otherwise, Start Canvas Pan
    setIsPanning(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
    onSelectNode(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    // 1. Handle Node Dragging
    if (draggedNode) {
        const deltaX = (e.clientX - draggedNode.startX) / zoom;
        const deltaY = (e.clientY - draggedNode.startY) / zoom;
        onNodeMove(draggedNode.id, draggedNode.initialNodeX + deltaX, draggedNode.initialNodeY + deltaY);
        return;
    }
    
    // 2. Handle Canvas Panning
    if (!isPanning) return;
    const dx = e.clientX - lastMousePos.x;
    const dy = e.clientY - lastMousePos.y;
    setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => { 
      setIsPanning(false); 
      setDraggedNode(null);
  };
  
  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    setZoom(Math.min(Math.max(0.2, zoom - e.deltaY * 0.0005), 2));
  };

  const renderEdges = () => {
    return edges.map(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);
      
      if (!sourceNode || !targetNode) return null;

      const sourceX = sourceNode.x + NODE_WIDTH; 
      const sourceY = sourceNode.y + 50; 
      const targetX = targetNode.x;
      const targetY = targetNode.y + 50;

      const dist = targetX - sourceX;
      const controlDist = Math.max(dist * 0.5, 100); 
      const path = `M ${sourceX} ${sourceY} C ${sourceX + controlDist} ${sourceY}, ${targetX - controlDist} ${targetY}, ${targetX} ${targetY}`;

      return (
        <g key={edge.id}>
            <path d={path} stroke="white" strokeWidth="6" fill="none" strokeOpacity="0.8" />
            <path d={path} stroke="#CBD5E1" strokeWidth="2" fill="none" className="transition-all duration-500" strokeLinecap="round" />
            <circle cx={targetX} cy={targetY} r="4" fill="white" stroke="#CBD5E1" strokeWidth="2" />
        </g>
      );
    });
  };

  return (
    <div 
      ref={containerRef}
      className={`w-full h-full overflow-hidden bg-[#F8FAFC] relative selection:bg-blue-100 ${isPanning ? 'cursor-grabbing' : 'cursor-grab'}`}
      onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onWheel={handleWheel}
    >
      <div style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`, transformOrigin: '0 0', width: '100%', height: '100%', position: 'absolute', top: 0, left: 0 }}>
        
        {/* Infinite Dot Background */}
        <div className="absolute top-[-5000px] bottom-[-5000px] left-[-5000px] w-[8000px] bg-dot opacity-60 pointer-events-none"></div>

        {/* LAB Label */}
        <div className="absolute top-[-250px] left-0 w-[1000px] flex justify-center pointer-events-none select-none">
          <div className="flex flex-col items-center opacity-20">
            <Microscope className="w-64 h-64 text-slate-400 mb-4" strokeWidth={0.5} />
            <div className="text-[8em] leading-none font-display font-bold text-slate-400 tracking-tighter">LAB</div>
          </div>
        </div>

        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-visible">{renderEdges()}</svg>

        {nodes.map(node => (
          <Node 
            key={node.id} 
            data={node} 
            selected={selectedNodeId === node.id}
            onClick={(e) => { e.stopPropagation(); onSelectNode(node.id); }}
            onAction={onNodeAction}
          />
        ))}
      </div>
      
      {/* Zoom Controls */}
      <div className="absolute bottom-8 right-8 flex flex-col gap-2 z-50">
         <div className="glass-panel p-1.5 rounded-xl flex flex-col gap-1 shadow-lg shadow-slate-200/50 scale-90 origin-bottom-right">
            <button onClick={() => setZoom(z => Math.min(z + 0.1, 2))} className="w-9 h-9 flex items-center justify-center text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors font-mono font-bold">+</button>
            <div className="h-px bg-slate-200 mx-2"></div>
            <button onClick={() => setZoom(z => Math.max(z - 0.1, 0.2))} className="w-9 h-9 flex items-center justify-center text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors font-mono font-bold">-</button>
         </div>
      </div>
    </div>
  );
});

export default Canvas;
