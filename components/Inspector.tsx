
import React, { useState, useRef } from 'react';
import { NodeData, AdCopy, ProjectContext, CampaignStage, AnalysisPhase } from '../types';
import { X, ThumbsUp, MessageCircle, Share2, Globe, MoreHorizontal, Download, Smartphone, Layout, Sparkles, BrainCircuit, Mic, Play, Pause, Wand2, ChevronLeft, ChevronRight, Layers, RefreshCw, Archive, Clock, ShieldAlert, BarChart3, AlertTriangle, Activity, CheckCircle2 } from 'lucide-react';
import { generateAdScript, generateVoiceover } from '../services/geminiService';

interface InspectorProps {
  node: NodeData;
  onClose: () => void;
  onAnalyze?: (nodeId: string) => void;
  onUpdate?: (id: string, data: Partial<NodeData>) => void;
  onRegenerate?: (id: string, aspectRatio: string) => void; 
  onPromote?: (id: string) => void; 
  project?: ProjectContext;
}

// Helper to decode raw PCM
const decodeAudioData = async (base64: string, ctx: AudioContext): Promise<AudioBuffer> => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
    const dataInt16 = new Int16Array(bytes.buffer);
    const frameCount = dataInt16.length;
    const buffer = ctx.createBuffer(1, frameCount, 24000);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < frameCount; i++) channelData[i] = dataInt16[i] / 32768.0;
    return buffer;
};

const Inspector: React.FC<InspectorProps> = ({ node, onClose, onAnalyze, onUpdate, onRegenerate, onPromote, project }) => {
  const { adCopy, imageUrl, carouselImages, title, format, postId, aiInsight, audioScript, audioBase64, stage, analysisPhase, metrics, testingTier, variableIsolated, congruenceRationale } = node;
  const [activeTab, setActiveTab] = useState<'PREVIEW' | 'INSIGHTS' | 'AUDIO'>('PREVIEW');
  const [aspectRatio, setAspectRatio] = useState<'SQUARE' | 'VERTICAL'>('SQUARE');
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [carouselIndex, setCarouselIndex] = useState(0); 
  
  const isLabAsset = stage === CampaignStage.TESTING;

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);

  // Carousel Logic
  const allImages = imageUrl ? [imageUrl, ...(carouselImages || [])] : [];
  const hasCarousel = allImages.length > 1;
  
  const handleNextSlide = () => {
    setCarouselIndex((prev) => (prev + 1) % allImages.length);
  };
  
  const handlePrevSlide = () => {
    setCarouselIndex((prev) => (prev - 1 + allImages.length) % allImages.length);
  };

  const handleGenerateScript = async () => {
    if (!onUpdate || !project) return;
    setIsGeneratingScript(true);
    const script = await generateAdScript(project, node.meta?.personaName || "User", node.title);
    onUpdate(node.id, { audioScript: script });
    setIsGeneratingScript(false);
  };

  const handleGenerateVoice = async () => {
    if (!onUpdate || !audioScript) return;
    setIsGeneratingAudio(true);
    const base64 = await generateVoiceover(audioScript, node.meta?.personaName || "User");
    if (base64) onUpdate(node.id, { audioBase64: base64 });
    setIsGeneratingAudio(false);
  };

  const handlePlayAudio = async () => {
    if (!audioBase64) return;
    if (isPlaying) { sourceRef.current?.stop(); setIsPlaying(false); return; }
    try {
        if (!audioContextRef.current) audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        const ctx = audioContextRef.current;
        const buffer = await decodeAudioData(audioBase64, ctx);
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        source.onended = () => setIsPlaying(false);
        sourceRef.current = source;
        source.start();
        setIsPlaying(true);
    } catch (e) { console.error("Audio Playback Error", e); setIsPlaying(false); }
  };

  const handleRegenerate = () => {
      if (onRegenerate) {
          onRegenerate(node.id, aspectRatio === 'SQUARE' ? '1:1' : '9:16');
      }
  };

  const handleDownload = () => {
      if (!imageUrl) return;
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = `${title.replace(/\s+/g, '_')}_${format.replace(/\s+/g, '_')}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  };

  const handlePromote = () => {
      if (onPromote) onPromote(node.id);
  };

  const getPhaseColor = (p?: AnalysisPhase) => {
      if (p === AnalysisPhase.PHASE_1) return 'text-slate-500 bg-slate-100 border-slate-200';
      if (p === AnalysisPhase.PHASE_2) return 'text-blue-600 bg-blue-50 border-blue-200';
      if (p === AnalysisPhase.PHASE_3) return 'text-purple-600 bg-purple-50 border-purple-200';
      if (p === AnalysisPhase.PHASE_4) return 'text-emerald-600 bg-emerald-50 border-emerald-200';
      return 'text-slate-500 bg-slate-100';
  };

  return (
    <div className="h-full w-full flex flex-col bg-white border-l border-slate-200 shadow-2xl animate-in slide-in-from-right duration-300">
      <div className="flex items-center justify-between p-4 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <h2 className="text-xs font-bold uppercase tracking-widest text-slate-500">Ad Inspector</h2>
        </div>
        <div className="flex items-center gap-2">
            {isLabAsset && onPromote && (
                 <button onClick={handlePromote} className="p-1.5 hover:bg-amber-100 text-slate-400 hover:text-amber-600 rounded-md transition-colors" title="Promote to Vault">
                     <Archive className="w-4 h-4" />
                 </button>
            )}
            <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-md text-slate-400 transition-colors"><X className="w-4 h-4" /></button>
        </div>
      </div>

      <div className="flex p-1 mx-4 mt-4 bg-slate-100 rounded-lg">
        <button onClick={() => setActiveTab('PREVIEW')} className={`flex-1 py-1.5 text-xs font-bold rounded-md flex items-center justify-center gap-2 transition-all ${activeTab === 'PREVIEW' ? 'bg-white shadow text-slate-900' : 'text-slate-500 hover:text-slate-700'}`}><Smartphone className="w-3.5 h-3.5" /> Preview</button>
        <button onClick={() => setActiveTab('AUDIO')} className={`flex-1 py-1.5 text-xs font-bold rounded-md flex items-center justify-center gap-2 transition-all ${activeTab === 'AUDIO' ? 'bg-white shadow text-pink-600' : 'text-slate-500 hover:text-slate-700'}`}><Mic className="w-3.5 h-3.5" /> Audio</button>
        <button onClick={() => setActiveTab('INSIGHTS')} className={`flex-1 py-1.5 text-xs font-bold rounded-md flex items-center justify-center gap-2 transition-all ${activeTab === 'INSIGHTS' ? 'bg-white shadow text-indigo-600' : 'text-slate-500 hover:text-slate-700'}`}><BrainCircuit className="w-3.5 h-3.5" /> Analysis</button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-slate-50/50">
        {activeTab === 'PREVIEW' && (
            <>
                <div className="flex justify-between items-center mb-6">
                    <div className="flex bg-white border border-slate-200 rounded-lg p-1 shadow-sm">
                        <button onClick={() => setAspectRatio('SQUARE')} className={`p-2 rounded transition-all ${aspectRatio === 'SQUARE' ? 'bg-slate-100 text-blue-600' : 'text-slate-400 hover:text-slate-600'}`} title="Feed (1:1)"><Layout className="w-4 h-4" /></button>
                        <button onClick={() => setAspectRatio('VERTICAL')} className={`p-2 rounded transition-all ${aspectRatio === 'VERTICAL' ? 'bg-slate-100 text-blue-600' : 'text-slate-400 hover:text-slate-600'}`} title="Story/Reels (9:16)"><Smartphone className="w-4 h-4" /></button>
                    </div>
                    {onRegenerate && (
                         <button onClick={handleRegenerate} disabled={node.isLoading} className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 text-blue-600 text-xs font-bold rounded-lg hover:bg-blue-100 transition-colors disabled:opacity-50">
                             <RefreshCw className={`w-3.5 h-3.5 ${node.isLoading ? 'animate-spin' : ''}`} />
                             Regenerate
                         </button>
                    )}
                </div>

                <div className={`bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mx-auto transition-all duration-300 ${aspectRatio === 'VERTICAL' ? 'max-w-[320px]' : 'max-w-[380px]'}`}>
                {aspectRatio === 'SQUARE' ? (
                    <>
                        <div className="p-3 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xs">Z</div>
                            <div className="flex flex-col leading-none"><span className="text-[13px] font-semibold text-slate-900">Zenith Focus</span><span className="text-[10px] text-slate-500">Sponsored · <Globe className="w-2.5 h-2.5 inline" /></span></div>
                            </div>
                            <MoreHorizontal className="w-4 h-4 text-slate-400" />
                        </div>
                        <div className="px-3 pb-2 text-[13px] text-slate-800 leading-relaxed whitespace-pre-wrap">
                            {adCopy ? adCopy.primaryText : <div className="space-y-2"><div className="h-2 bg-slate-100 rounded w-full animate-pulse"/><div className="h-2 bg-slate-100 rounded w-3/4 animate-pulse"/></div>}
                        </div>
                        
                        <div className="aspect-square bg-slate-100 relative group cursor-pointer overflow-hidden">
                            {allImages[carouselIndex] ? (
                                <img src={allImages[carouselIndex]} alt={`Slide ${carouselIndex + 1}`} className="w-full h-full object-cover transition-all duration-300" />
                            ) : <div className="w-full h-full flex items-center justify-center text-slate-300">No Image</div>}
                            
                            {/* Carousel UI Controls */}
                            {hasCarousel && (
                                <>
                                    <button onClick={(e) => {e.stopPropagation(); handlePrevSlide();}} className="absolute left-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-white/80 hover:bg-white rounded-full flex items-center justify-center shadow-sm opacity-0 group-hover:opacity-100 transition-opacity text-slate-800"><ChevronLeft className="w-4 h-4"/></button>
                                    <button onClick={(e) => {e.stopPropagation(); handleNextSlide();}} className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-white/80 hover:bg-white rounded-full flex items-center justify-center shadow-sm opacity-0 group-hover:opacity-100 transition-opacity text-slate-800"><ChevronRight className="w-4 h-4"/></button>
                                    <div className="absolute top-3 right-3 bg-black/50 backdrop-blur-sm px-2 py-1 rounded-full text-[10px] font-bold text-white flex items-center gap-1">
                                        <Layers className="w-3 h-3" /> {carouselIndex + 1}/{allImages.length}
                                    </div>
                                    <div className="absolute bottom-3 left-0 right-0 flex justify-center gap-1.5">
                                        {allImages.map((_, idx) => (
                                            <div key={idx} className={`w-1.5 h-1.5 rounded-full shadow-sm transition-all ${idx === carouselIndex ? 'bg-blue-500 scale-125' : 'bg-white/70'}`}></div>
                                        ))}
                                    </div>
                                </>
                            )}
                        </div>

                        <div className="bg-slate-50 p-3 flex items-center justify-between border-t border-slate-100">
                            <div className="flex flex-col max-w-[200px]">
                                <span className="text-[10px] text-slate-500 uppercase truncate">zenithfocus.com</span>
                                <span className="text-[14px] font-bold text-slate-900 leading-tight line-clamp-1">{adCopy ? adCopy.headline : title}</span>
                            </div>
                            <button className="px-4 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-800 text-[12px] font-semibold rounded transition-colors">{adCopy?.cta || "Learn More"}</button>
                        </div>
                        <div className="px-3 py-2 border-t border-slate-100 flex items-center justify-between text-slate-500">
                            <div className="flex items-center gap-1 text-xs"><ThumbsUp className="w-3 h-3" /> <span>244</span></div>
                            <div className="flex items-center gap-4 text-xs"><span>42 Comments</span><span>12 Shares</span></div>
                        </div>
                    </>
                ) : (
                    <div className="aspect-[9/16] relative bg-slate-900 text-white overflow-hidden">
                         {allImages[carouselIndex] ? <img src={allImages[carouselIndex]} alt="Ad Creative" className="w-full h-full object-cover opacity-90" /> : <div className="w-full h-full flex items-center justify-center text-slate-600">No Image</div>}
                         <div className="absolute top-4 left-4 flex items-center gap-2"><div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xs border-2 border-white">Z</div><span className="text-sm font-semibold shadow-black drop-shadow-md">Zenith Focus</span></div>
                         <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent pt-20">
                            <div className="mb-4"><p className="text-sm font-medium leading-snug drop-shadow-md line-clamp-3">{adCopy?.primaryText}</p></div>
                            <button className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg flex items-center justify-center gap-2 transition-colors">{adCopy?.cta || "Shop Now"} <MoreHorizontal className="w-4 h-4 rotate-90" /></button>
                         </div>
                         {hasCarousel && (
                            <div className="absolute bottom-[140px] left-0 right-0 flex justify-center gap-1">
                                {allImages.map((_, idx) => <div key={idx} className={`h-1 rounded-full shadow-sm transition-all ${idx === carouselIndex ? 'bg-white w-6' : 'bg-white/50 w-2'}`}></div>)}
                            </div>
                         )}
                    </div>
                )}
                </div>
            </>
        )}

        {activeTab === 'AUDIO' && (
            <div className="space-y-6">
                <div className="p-4 bg-pink-50 border border-pink-100 rounded-xl">
                    <h3 className="text-sm font-bold text-pink-900 mb-2 flex items-center gap-2"><Mic className="w-4 h-4" /> UGC Script Generator</h3>
                    <p className="text-xs text-pink-800/80 mb-4">Generate a TikTok-style script and use AI Voiceover to bring it to life.</p>
                    {audioScript ? <div className="bg-white p-3 rounded-lg border border-pink-100 mb-4"><p className="text-sm text-slate-700 whitespace-pre-wrap font-mono leading-relaxed">{audioScript}</p></div> : <button onClick={handleGenerateScript} disabled={isGeneratingScript} className="w-full py-2 bg-white border border-pink-200 text-pink-700 text-xs font-bold rounded-lg hover:bg-pink-100 transition-colors flex items-center justify-center gap-2">{isGeneratingScript ? <div className="w-3 h-3 rounded-full border-2 border-pink-600 border-t-transparent animate-spin"/> : <Wand2 className="w-3.5 h-3.5" />} Generate Script</button>}
                    {audioScript && (<div className="flex gap-2 mt-2"><button onClick={handleGenerateScript} className="flex-1 py-2 text-xs text-pink-600 hover:bg-pink-100 rounded-lg transition-colors">Regenerate</button>{!audioBase64 && (<button onClick={handleGenerateVoice} disabled={isGeneratingAudio} className="flex-1 py-2 bg-pink-600 text-white text-xs font-bold rounded-lg hover:bg-pink-700 transition-colors flex items-center justify-center gap-2">{isGeneratingAudio ? <div className="w-3 h-3 rounded-full border-2 border-white border-t-transparent animate-spin"/> : <Mic className="w-3.5 h-3.5" />} Synthesize Voice</button>)}</div>)}
                </div>
                {audioBase64 && (<div className="p-4 bg-slate-900 rounded-xl text-white"><div className="flex items-center justify-between mb-4"><div className="flex items-center gap-3"><div className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center"><div className={`space-x-0.5 flex items-end h-4 ${isPlaying ? 'animate-pulse' : ''}`}><div className="w-1 bg-pink-500 h-2 rounded-full" /><div className="w-1 bg-pink-500 h-4 rounded-full" /><div className="w-1 bg-pink-500 h-3 rounded-full" /></div></div><div><h4 className="text-sm font-bold">AI Voiceover</h4><p className="text-xs text-slate-400">Gemini 2.5 Flash TTS</p></div></div></div><button onClick={handlePlayAudio} className="w-full py-3 bg-pink-600 hover:bg-pink-500 rounded-lg font-bold flex items-center justify-center gap-2 transition-colors">{isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />} {isPlaying ? 'Stop Audio' : 'Play Voiceover'}</button></div>)}
            </div>
        )}

        {activeTab === 'INSIGHTS' && (
            <div className="space-y-6">
                 {/* ANDROMEDA DASHBOARD */}
                 {node.metrics ? (
                     <>
                        <div className={`p-4 rounded-xl border ${getPhaseColor(analysisPhase)} shadow-sm`}>
                             <div className="flex items-center justify-between mb-2">
                                <span className="text-[10px] uppercase font-bold tracking-widest flex items-center gap-2">
                                    <Clock className="w-3 h-3" /> Ad Age: {node.metrics.ageHours} hrs
                                </span>
                                <span className="text-[10px] px-2 py-0.5 bg-white/50 rounded border border-black/5 font-bold">{analysisPhase}</span>
                             </div>
                             
                             {analysisPhase === AnalysisPhase.PHASE_1 && (
                                 <div className="text-xs font-medium flex items-center gap-2 mt-2">
                                     <ShieldAlert className="w-4 h-4" />
                                     <span>Volatility High. Do Not Touch. 72hr Rule Active.</span>
                                 </div>
                             )}
                             {analysisPhase === AnalysisPhase.PHASE_2 && (
                                 <div className="text-xs font-medium flex items-center gap-2 mt-2">
                                     <Activity className="w-4 h-4" />
                                     <span>Health Check Mode. Monitor CTR & CPM.</span>
                                 </div>
                             )}
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div className="p-3 bg-white rounded-xl border border-slate-200 shadow-sm opacity-60"><span className="text-[10px] uppercase text-slate-400 font-bold">Spend</span><div className="text-2xl font-mono font-bold text-slate-700">${node.metrics.spend}</div></div>
                            <div className="p-3 bg-white rounded-xl border border-slate-200 shadow-sm"><span className="text-[10px] uppercase text-slate-400 font-bold">CTR (Thumbstop)</span><div className="text-2xl font-mono font-bold text-blue-500">{node.metrics.ctr.toFixed(2)}%</div></div>
                            
                            {/* Hide ROAS/CPA in Phase 1 */}
                            {analysisPhase !== AnalysisPhase.PHASE_1 ? (
                                <>
                                    <div className="p-3 bg-white rounded-xl border border-slate-200 shadow-sm"><span className="text-[10px] uppercase text-slate-400 font-bold">ROAS</span><div className={`text-2xl font-mono font-bold ${node.metrics.roas > 2 ? 'text-emerald-500' : 'text-slate-700'}`}>{node.metrics.roas.toFixed(2)}x</div></div>
                                    <div className="p-3 bg-white rounded-xl border border-slate-200 shadow-sm"><span className="text-[10px] uppercase text-slate-400 font-bold">CPA</span><div className="text-2xl font-mono font-bold text-slate-700">${node.metrics.cpa.toFixed(0)}</div></div>
                                </>
                            ) : (
                                <div className="col-span-2 p-3 bg-slate-50 border border-slate-200 rounded-xl flex items-center justify-center gap-2 text-slate-400 text-xs font-mono">
                                    <AlertTriangle className="w-4 h-4" /> ROAS HIDDEN DURING LEARNING PHASE
                                </div>
                            )}
                        </div>
                     </>
                 ) : (<div className="p-4 bg-amber-50 border border-amber-100 rounded-xl text-amber-800 text-xs">No performance data available yet. Run a simulation first.</div>)}
                 
                 {/* STRATEGIC ANALYSIS */}
                 <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-4">
                     <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2"><Sparkles className="w-4 h-4 text-indigo-600" /><h3 className="text-sm font-bold text-indigo-900">Strategic Analysis</h3></div>
                        {testingTier && <span className="text-[9px] bg-indigo-200 text-indigo-800 px-1.5 py-0.5 rounded font-bold">{testingTier}</span>}
                     </div>
                     
                     {congruenceRationale && (
                         <div className="mb-3 pb-3 border-b border-indigo-200/50">
                             <span className="text-[10px] text-indigo-500 font-bold uppercase block mb-1 flex items-center gap-1"><CheckCircle2 className="w-3 h-3"/> Congruency Check</span>
                             <p className="text-xs text-indigo-900/80 font-medium bg-white/50 p-2 rounded-lg border border-indigo-100">{congruenceRationale}</p>
                         </div>
                     )}

                     {variableIsolated && (
                         <div className="mb-3 pb-3 border-b border-indigo-200/50">
                             <span className="text-[10px] text-indigo-400 font-bold uppercase block mb-1">Variable Isolated</span>
                             <p className="text-xs text-indigo-800 font-medium">{variableIsolated}</p>
                         </div>
                     )}

                     {aiInsight ? (<p className="text-sm text-indigo-800 leading-relaxed">{aiInsight}</p>) : (<div className="text-center py-6"><p className="text-xs text-indigo-400 mb-3">Unlock AI insights for this creative.</p><button onClick={() => onAnalyze && onAnalyze(node.id)} disabled={!node.metrics} className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed">Generate Analysis</button></div>)}
                 </div>
            </div>
        )}
      </div>

      <div className="p-4 border-t border-slate-200 bg-white">
        <button onClick={handleDownload} className="w-full py-3 bg-slate-900 hover:bg-black text-white rounded-xl font-bold flex items-center justify-center gap-2 transition-colors"><Download className="w-4 h-4" /> Download Assets</button>
      </div>
    </div>
  );
};

export default Inspector;
