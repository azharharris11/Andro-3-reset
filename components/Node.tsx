
import React from 'react';
import { NodeData, NodeType, CampaignStage } from '../types';
import { User, Zap, Image as ImageIcon, Target, Award, RefreshCw, Sparkles, TrendingUp, DollarSign, MousePointer2, Ghost, Mic, Layers, Cpu, Archive } from 'lucide-react';

interface NodeProps {
  data: NodeData;
  selected: boolean;
  onClick: (e: React.MouseEvent) => void;
  onAction: (action: string, id: string) => void;
  isGridView?: boolean; 
}

const Node: React.FC<NodeProps> = ({ data, selected, onClick, onAction, isGridView = false }) => {
  
  const isScaling = data.stage === CampaignStage.SCALING;
  const isFatigued = isScaling && data.metrics && data.metrics.roas < 1.5;
  const hasParent = !!data.parentId && !isGridView;
  const showOutputHandle = !isGridView && (data.type !== NodeType.CREATIVE || (data.type === NodeType.CREATIVE && isScaling));
  const isGhost = data.isGhost;
  const isCarousel = data.carouselImages && data.carouselImages.length > 0;

  const getStatusStyles = () => {
    if (isGhost) return {
        container: 'bg-slate-50/30 ring-1 ring-slate-200 border border-dashed border-slate-300 opacity-60 grayscale',
        header: 'bg-slate-100/50 border-b border-slate-200 text-slate-400',
        text: 'text-slate-400',
        accent: 'text-slate-400',
        iconBg: 'bg-slate-100',
        handle: 'bg-slate-200'
    };
    if (isFatigued) return {
      container: 'bg-white ring-1 ring-red-200 shadow-xl shadow-red-500/5',
      header: 'bg-red-50/50 border-b border-red-100 text-red-700',
      text: 'text-slate-800',
      accent: 'text-red-600',
      iconBg: 'bg-red-100',
      handle: 'bg-red-400'
    };
    if (isScaling) return {
      container: 'bg-white ring-1 ring-amber-200 shadow-xl shadow-amber-500/5',
      header: 'bg-amber-50/50 border-b border-amber-100 text-amber-800',
      text: 'text-amber-950',
      accent: 'text-amber-600',
      iconBg: 'bg-amber-100',
      handle: 'bg-amber-400'
    };
    if (data.isWinning) return {
      container: 'bg-white ring-1 ring-emerald-200 shadow-xl shadow-emerald-500/5',
      header: 'bg-emerald-50/50 border-b border-emerald-100 text-emerald-700',
      text: 'text-slate-800',
      accent: 'text-emerald-600',
      iconBg: 'bg-emerald-100',
      handle: 'bg-emerald-400'
    };
    if (selected) return {
      container: 'bg-white ring-2 ring-blue-600 shadow-2xl shadow-blue-500/20',
      header: 'bg-blue-50/50 border-b border-blue-100 text-blue-700',
      text: 'text-slate-900',
      accent: 'text-blue-600',
      iconBg: 'bg-blue-100',
      handle: 'bg-blue-500'
    };
    
    return {
      container: 'bg-white ring-1 ring-slate-200 shadow-md hover:shadow-xl', 
      header: 'bg-slate-50/50 border-b border-slate-100 text-slate-500',
      text: 'text-slate-800',
      accent: 'text-slate-500',
      iconBg: 'bg-slate-100',
      handle: 'bg-slate-300'
    };
  };

  const styles = getStatusStyles();

  const getIcon = () => {
    if (isGhost) return <Ghost className={`w-3.5 h-3.5 ${styles.accent}`} />;
    if (isScaling) return <Award className={`w-3.5 h-3.5 ${styles.accent}`} />;
    switch (data.type) {
      case NodeType.ROOT: return <Zap className="w-3.5 h-3.5 text-purple-600" />;
      case NodeType.PERSONA: return <User className="w-3.5 h-3.5 text-teal-600" />;
      case NodeType.ANGLE: return <Target className="w-3.5 h-3.5 text-pink-600" />;
      case NodeType.CREATIVE: return <ImageIcon className="w-3.5 h-3.5 text-indigo-600" />;
    }
  };

  const formatTokens = (input: number = 0, output: number = 0) => {
    const total = input + output;
    if (total > 1000) return `${(total / 1000).toFixed(1)}k`;
    return total;
  };

  const containerClass = isGridView 
    ? `relative w-full h-full flex flex-col rounded-xl border-0 transition-all duration-300 ${styles.container} ${selected ? 'ring-2 ring-amber-400' : ''}`
    : `absolute w-[320px] rounded-xl border-0 node-interactive node-enter flex flex-col ${styles.container} ${selected ? 'z-50 scale-[1.02]' : 'z-10'}`;

  const styleProp = isGridView ? {} : { left: data.x, top: data.y, transition: 'box-shadow 0.2s, transform 0.2s' };

  return (
    <div
      onClick={onClick}
      className={containerClass}
      style={styleProp}
      data-id={data.id} 
    >
      {!isGridView && !isGhost && (
        <>
          {hasParent && (
            <div className="absolute left-0 top-[50px] -translate-x-1/2 -translate-y-1/2 w-4 h-4 flex items-center justify-center z-[-1]">
                <div className={`w-2.5 h-2.5 rounded-full border-2 border-white shadow-sm node-handle ${styles.handle}`}></div>
            </div>
          )}
          {showOutputHandle && (
            <div className="absolute right-0 top-[50px] translate-x-1/2 -translate-y-1/2 w-4 h-4 flex items-center justify-center z-[-1]">
                <div className={`w-2.5 h-2.5 rounded-full border-2 border-white shadow-sm node-handle ${styles.handle}`}></div>
            </div>
          )}
        </>
      )}

      <div className={`h-10 flex items-center justify-between px-4 rounded-t-xl ${styles.header}`}>
        <div className="flex items-center gap-2.5">
          <div className={`p-1 rounded-md ${styles.iconBg}`}>
             {getIcon()}
          </div>
          <span className="text-[10px] font-bold uppercase tracking-widest font-mono opacity-80">
             {isGhost ? 'ARCHIVED' : (isScaling ? 'VAULT ASSET' : data.type)}
          </span>
        </div>
        <div className="flex items-center gap-2">
            {data.isLoading && <div className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />}
            
            {(data.inputTokens !== undefined || data.outputTokens !== undefined) && (
                 <div className="flex items-center gap-1 bg-slate-200/50 px-1.5 py-0.5 rounded border border-slate-300/50" title={`Input: ${data.inputTokens} | Output: ${data.outputTokens}`}>
                    <Cpu className="w-2.5 h-2.5 text-slate-500" />
                    <span className="text-[9px] font-mono text-slate-700 font-bold">{formatTokens(data.inputTokens, data.outputTokens)}</span>
                 </div>
            )}

            {data.estimatedCost !== undefined && (
                <div className="flex items-center gap-1 bg-green-100/50 px-1.5 py-0.5 rounded border border-green-200/50">
                    <span className="text-[9px] font-mono text-green-700 font-bold">~${data.estimatedCost.toFixed(2)}</span>
                </div>
            )}
            
            {data.audioBase64 && <div className="w-4 h-4 rounded-full bg-pink-100 flex items-center justify-center"><Mic className="w-2.5 h-2.5 text-pink-600" /></div>}
            {isCarousel && <div className="w-4 h-4 rounded-full bg-blue-100 flex items-center justify-center" title="Carousel Format"><Layers className="w-2.5 h-2.5 text-blue-600" /></div>}
            {isFatigued && <span className="text-[9px] bg-red-100 text-red-600 px-2 py-0.5 rounded-full border border-red-200 font-bold animate-pulse tracking-wide">FATIGUE</span>}
        </div>
      </div>

      <div className={`p-4 flex flex-col gap-3 bg-white/50 rounded-b-xl flex-1 ${isGhost ? 'opacity-50' : ''}`}>
        
        {data.type === NodeType.CREATIVE && data.imageUrl && (
           <div className="relative w-full aspect-square rounded-lg overflow-hidden border border-slate-100 group shadow-sm bg-slate-50 select-none pointer-events-none">
             <img src={data.imageUrl} alt="Creative" className="w-full h-full object-cover mix-blend-multiply" />
             
             {isCarousel && (
                <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-md text-white p-1 rounded-md">
                    <Layers className="w-3 h-3" />
                </div>
             )}

             <div className="absolute bottom-2 left-2 right-2 flex justify-between">
                <span className="inline-block px-2 py-1 bg-white/90 backdrop-blur-sm border border-slate-100 shadow-sm rounded text-[10px] text-slate-800 font-medium truncate max-w-[180px]">
                    {data.format}
                </span>
             </div>
           </div>
        )}

        <div>
          <h3 className={`text-sm font-display font-semibold leading-snug ${styles.text}`}>
            {data.title}
          </h3>
          {data.description && !isGridView && (
            <p className="mt-2 text-xs text-slate-500 leading-relaxed font-light border-l-2 border-slate-100 pl-2 line-clamp-2">
              {data.description}
            </p>
          )}
        </div>

        {data.metrics && (
            <div className="grid grid-cols-3 gap-px bg-slate-100 border border-slate-100 rounded-lg overflow-hidden mt-auto">
                <div className="p-2 flex flex-col gap-0.5 items-center justify-center bg-white group">
                    <span className="text-[8px] uppercase text-slate-400 font-bold group-hover:text-amber-500 flex items-center gap-1"><DollarSign className="w-2 h-2"/> Spend</span>
                    <span className="text-[11px] font-mono text-slate-900 font-medium">${data.metrics.spend.toLocaleString()}</span>
                </div>
                <div className="p-2 flex flex-col gap-0.5 items-center justify-center bg-white group">
                    <span className="text-[8px] uppercase text-slate-400 font-bold group-hover:text-emerald-500 flex items-center gap-1"><TrendingUp className="w-2 h-2"/> ROAS</span>
                    <span className={`text-[11px] font-mono font-bold ${data.metrics.roas > 2.5 ? 'text-emerald-600' : isFatigued ? 'text-red-500' : 'text-slate-900'}`}>
                        {data.metrics.roas.toFixed(2)}x
                    </span>
                </div>
                <div className="p-2 flex flex-col gap-0.5 items-center justify-center bg-white group">
                    <span className="text-[8px] uppercase text-slate-400 font-bold group-hover:text-blue-500 flex items-center gap-1"><MousePointer2 className="w-2 h-2"/> CPA</span>
                    <span className="text-[11px] font-mono text-slate-900 font-medium">${data.metrics.cpa.toFixed(0)}</span>
                </div>
            </div>
        )}
      </div>

      <div className="p-1.5 bg-slate-50/80 border-t border-slate-100 rounded-b-xl">
        {!isGridView && !data.metrics && !isGhost && (
            <>
            {data.type === NodeType.ROOT && (
                <button 
                onClick={(e) => { e.stopPropagation(); onAction('expand_personas', data.id); }}
                className="w-full py-2 bg-white hover:bg-blue-50 hover:border-blue-200 text-blue-600 text-xs font-medium rounded-lg border border-slate-200 transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                <User className="w-3 h-3" /> Branch Personas
                </button>
            )}
            {data.type === NodeType.PERSONA && (
                <button 
                onClick={(e) => { e.stopPropagation(); onAction('expand_angles', data.id); }}
                className="w-full py-2 bg-white hover:bg-pink-50 hover:border-pink-200 text-pink-600 text-xs font-medium rounded-lg border border-slate-200 transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                <Target className="w-3 h-3" /> Expand Angles
                </button>
            )}
            {data.type === NodeType.ANGLE && (
                <button 
                onClick={(e) => { e.stopPropagation(); onAction('generate_creatives', data.id); }}
                className="w-full py-2 bg-white hover:bg-indigo-50 hover:border-indigo-200 text-indigo-600 text-xs font-medium rounded-lg border border-slate-200 transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                <Sparkles className="w-3 h-3" /> Generate Formats
                </button>
            )}
            </>
        )}
        
        {/* MANUAL PROMOTION ALLOWED FOR ALL CREATIVES */}
        {!isGridView && data.type === NodeType.CREATIVE && !isScaling && !isGhost && (
            <button 
            onClick={(e) => { e.stopPropagation(); onAction('promote_creative', data.id); }}
            className={`w-full py-2 text-xs font-bold rounded-lg shadow-sm transition-all flex items-center justify-center gap-2 ${
                data.isWinning 
                    ? 'bg-amber-400 hover:bg-amber-300 text-amber-950 shadow-amber-200' // Gold for Winners
                    : 'bg-white hover:bg-slate-50 border border-slate-200 text-slate-600 hover:text-blue-600 hover:border-blue-200' // Neutral for Manual
            }`}
            >
            {data.isWinning ? <Award className="w-3.5 h-3.5" /> : <Archive className="w-3.5 h-3.5" />} 
            {data.isWinning ? 'PROMOTE WINNER' : 'MOVE TO VAULT'}
            </button>
        )}

        {isGhost && (
            <div className="w-full py-2 text-xs text-center text-slate-400 font-mono">
                SCALED TO VAULT
            </div>
        )}
        {isScaling && !data.isLoading && (
            <button 
            onClick={(e) => { e.stopPropagation(); onAction('remix_creative', data.id); }}
            className={`w-full py-2 text-xs font-bold rounded-lg transition-all flex items-center justify-center gap-2 border shadow-sm ${
                isFatigued 
                ? 'bg-red-500 hover:bg-red-400 text-white border-red-600 animate-pulse shadow-red-200' 
                : 'bg-white hover:bg-slate-50 text-slate-600 border-slate-200'
            }`}
            >
            <RefreshCw className="w-3.5 h-3.5" />
            {isFatigued ? 'FATIGUE - REMIX NOW' : 'Create Variations'}
            </button>
        )}
      </div>
    </div>
  );
};

export default Node;
