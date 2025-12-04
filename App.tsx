import React, { useState, useRef } from 'react';
import { HashRouter } from 'react-router-dom';
import { Layers, Settings, Activity, Microscope, ShieldCheck, X, RefreshCw, Globe, Sparkles, Image as ImageIcon, Upload, Package, Megaphone, Filter, Target, FileText, MapPin, Info, Smartphone } from 'lucide-react';
import Canvas, { CanvasHandle } from './components/Canvas';
import Node from './components/Node';
import Inspector from './components/Inspector';
import { NodeType, NodeData, Edge, ProjectContext, CreativeFormat, CampaignStage, ViewMode, FunnelStage, MarketAwareness, CopyFramework, AnalysisPhase } from './types';
import { generatePersonas, generateAngles, generateCreativeImage, generateAdCopy, generateCarouselSlides, generateCreativeConcept, checkAdCompliance, analyzeLandingPageContext, analyzeImageContext } from './services/geminiService';
import { scrapeLandingPage } from './services/firecrawlService';

const INITIAL_PROJECT: ProjectContext = {
  productName: "Zenith Focus Gummies",
  productDescription: "Nootropic gummies for focus and memory without the caffeine crash.",
  targetAudience: "Students, Programmers, and Creatives.",
  targetCountry: "USA",
  brandVoice: "Witty, Smart, but Approachable",
  funnelStage: FunnelStage.TOF,
  marketAwareness: MarketAwareness.PROBLEM_AWARE,
  copyFramework: CopyFramework.PAS,
  offer: "Buy 2 Get 1 Free"
};

const FORMAT_GROUPS: Record<string, CreativeFormat[]> = {
  "Carousel Specials (High Engagement)": [
    CreativeFormat.CAROUSEL_REAL_STORY, // NEW
    CreativeFormat.CAROUSEL_EDUCATIONAL,
    CreativeFormat.CAROUSEL_TESTIMONIAL,
    CreativeFormat.CAROUSEL_PANORAMA,
    CreativeFormat.CAROUSEL_PHOTO_DUMP,
  ],
  "Instagram Native": [
    CreativeFormat.STORY_QNA, 
    CreativeFormat.STORY_POLL,
    CreativeFormat.REELS_THUMBNAIL,
    CreativeFormat.DM_NOTIFICATION,
    CreativeFormat.UGC_MIRROR,
    CreativeFormat.PHONE_NOTES,
    CreativeFormat.TWITTER_REPOST,
  ],
  "Direct Response Winners": [
    CreativeFormat.BENEFIT_POINTERS, // NEW
    CreativeFormat.SOCIAL_COMMENT_STACK, // NEW
    CreativeFormat.STICKY_NOTE_REALISM, // NEW
    CreativeFormat.HANDHELD_TWEET, // NEW
  ],
  "Logic & Rational": [
    CreativeFormat.US_VS_THEM,
    CreativeFormat.GRAPH_CHART,
    CreativeFormat.TIMELINE_JOURNEY,
  ],
  "Social Proof & Voyeurism": [
    CreativeFormat.CHAT_CONVERSATION,
    CreativeFormat.REMINDER_NOTIF,
  ],
  "Product Centric": [
    CreativeFormat.POV_HANDS,
    CreativeFormat.ANNOTATED_PRODUCT,
    CreativeFormat.SEARCH_BAR,
  ],
  "Aesthetic & Mood": [
    CreativeFormat.COLLAGE_SCRAPBOOK,
    CreativeFormat.CHECKLIST_TODO,
    CreativeFormat.AESTHETIC_MINIMAL,
  ],
  "Pattern Interrupts": [
    CreativeFormat.BIG_FONT,
    CreativeFormat.GMAIL_UX,
    CreativeFormat.UGLY_VISUAL,
    CreativeFormat.MS_PAINT,
    CreativeFormat.MEME,
    CreativeFormat.LONG_TEXT,
    CreativeFormat.BEFORE_AFTER,
    CreativeFormat.CARTOON,
    CreativeFormat.WHITEBOARD,
    CreativeFormat.REDDIT_THREAD
  ]
};

const App = () => {
  const [project, setProject] = useState<ProjectContext>(INITIAL_PROJECT);
  const [activeView, setActiveView] = useState<ViewMode>('LAB');
  
  const [nodes, setNodes] = useState<NodeData[]>([
    {
      id: 'root',
      type: NodeType.ROOT,
      title: INITIAL_PROJECT.productName,
      description: INITIAL_PROJECT.productDescription,
      x: 0,
      y: 0,
      stage: CampaignStage.TESTING
    }
  ]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [isConfigOpen, setIsConfigOpen] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isFormatModalOpen, setIsFormatModalOpen] = useState(false);
  const [targetAngleId, setTargetAngleId] = useState<string | null>(null);
  const [selectedFormats, setSelectedFormats] = useState<Set<CreativeFormat>>(new Set());
  
  // New States for Firecrawl & Image Analysis
  const [landingPageUrl, setLandingPageUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAnalyzingImage, setIsAnalyzingImage] = useState(false);
  
  const canvasRef = useRef<CanvasHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const productRefInputRef = useRef<HTMLInputElement>(null);

  const labNodes = nodes.filter(n => n.stage === CampaignStage.TESTING || n.isGhost);
  const labEdges = edges.filter(e => {
      const source = nodes.find(n => n.id === e.source);
      const target = nodes.find(n => n.id === e.target);
      return (source?.stage === CampaignStage.TESTING || source?.isGhost) && 
             (target?.stage === CampaignStage.TESTING || target?.isGhost);
  });
  const vaultNodes = nodes.filter(n => n.stage === CampaignStage.SCALING);
  const selectedNode = nodes.find(n => n.id === selectedNodeId);

  const addNode = (node: NodeData) => { setNodes(prev => [...prev, node]); };
  const addEdge = (source: string, target: string) => { setEdges(prev => [...prev, { id: `${source}-${target}`, source, target }]); };
  const updateNode = (id: string, updates: Partial<NodeData>) => { setNodes(prev => prev.map(n => n.id === id ? { ...n, ...updates } : n)); };
  
  // Handle dragging nodes
  const handleNodeMove = (id: string, x: number, y: number) => {
      setNodes(prev => prev.map(n => n.id === id ? { ...n, x, y } : n));
  };

  // --- LOGIC: MARKET AWARENESS -> FUNNEL STAGE ---
  const handleAwarenessChange = (awareness: MarketAwareness) => {
    let derivedFunnelStage = FunnelStage.TOF;

    switch (awareness) {
      case MarketAwareness.UNAWARE:
      case MarketAwareness.PROBLEM_AWARE:
        derivedFunnelStage = FunnelStage.TOF;
        break;
      case MarketAwareness.SOLUTION_AWARE:
        derivedFunnelStage = FunnelStage.MOF;
        break;
      case MarketAwareness.PRODUCT_AWARE:
      case MarketAwareness.MOST_AWARE:
        derivedFunnelStage = FunnelStage.BOF;
        break;
    }

    setProject(prev => ({
      ...prev,
      marketAwareness: awareness,
      funnelStage: derivedFunnelStage
    }));
  };

  // --- FIRECRAWL ANALYSIS ---
  const handleAnalyzeUrl = async () => {
      if (!landingPageUrl) return;
      
      setIsAnalyzing(true);
      try {
          // 1. Scrape with Firecrawl
          const scrapeResult = await scrapeLandingPage(landingPageUrl);
          
          if (!scrapeResult.success || !scrapeResult.markdown) {
              alert("Failed to read the website. Please enter details manually.");
              setIsAnalyzing(false);
              return;
          }

          // 2. Analyze with Gemini
          const context = await analyzeLandingPageContext(scrapeResult.markdown);
          
          setProject({
              ...project,
              productName: context.productName,
              productDescription: context.productDescription,
              targetAudience: context.targetAudience,
              targetCountry: context.targetCountry || project.targetCountry,
              landingPageUrl: landingPageUrl
          });

          // Update Root Node
          setNodes(prev => prev.map(n => n.type === NodeType.ROOT ? {
              ...n,
              title: context.productName,
              description: context.productDescription
          } : n));

      } catch (e) {
          console.error(e);
          alert("Analysis failed. Please check the URL and try again.");
      }
      setIsAnalyzing(false);
  };

  // --- IMAGE ANALYSIS ---
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      setIsAnalyzingImage(true);
      
      const reader = new FileReader();
      reader.onloadend = async () => {
          const base64 = reader.result as string;
          try {
              const context = await analyzeImageContext(base64);
              
              setProject({
                  ...project, // Keep url if exists
                  productName: context.productName,
                  productDescription: context.productDescription,
                  targetAudience: context.targetAudience,
                  targetCountry: context.targetCountry || project.targetCountry
              });

               // Update Root Node
              setNodes(prev => prev.map(n => n.type === NodeType.ROOT ? {
                  ...n,
                  title: context.productName,
                  description: context.productDescription
              } : n));

          } catch (error) {
              console.error(error);
              alert("Could not analyze image. Try a clearer product shot.");
          }
          setIsAnalyzingImage(false);
      };
      reader.readAsDataURL(file);
  };

  // --- PRODUCT REFERENCE IMAGE ---
  const handleProductRefUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onloadend = () => {
          const base64 = reader.result as string;
          setProject(prev => ({ ...prev, productReferenceImage: base64 }));
      };
      reader.readAsDataURL(file);
  };

  // New: Handle Regeneration from Inspector
  const handleRegenerateNode = async (nodeId: string, aspectRatio: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    updateNode(nodeId, { isLoading: true, description: "Regenerating visual..." });

    try {
        const personaName = node.meta?.personaName || "User";
        const angle = node.meta?.angle || node.title;
        // Fallback for legacy nodes that only had styleContext (visualScene)
        const visualScene = node.meta?.visualScene || node.meta?.styleContext || ""; 
        const visualStyle = node.meta?.visualStyle || "";
        const technicalPrompt = node.meta?.technicalPrompt || "";
        const format = node.format as CreativeFormat;

        const imgResult = await generateCreativeImage(
            project, personaName, angle, format, 
            visualScene, visualStyle, technicalPrompt, 
            aspectRatio
        );
        
        if (imgResult.data) {
            updateNode(nodeId, { 
                imageUrl: imgResult.data,
                isLoading: false,
                description: node.adCopy?.primaryText.slice(0, 100) + "..." || node.description
            });
        } else {
            updateNode(nodeId, { isLoading: false, description: "Regeneration failed." });
        }
    } catch (e) {
        console.error("Regeneration failed", e);
        updateNode(nodeId, { isLoading: false, description: "Error during regeneration" });
    }
  };

  const executeGeneration = async (nodeId: string, formats: CreativeFormat[]) => {
    const parentNode = nodes.find(n => n.id === nodeId);
    if (!parentNode) return;

    updateNode(nodeId, { isLoading: true });

    // Grid Layout Constants
    const HORIZONTAL_GAP = 550; 
    const COL_SPACING = 350;    
    const ROW_SPACING = 400;    
    const COLUMNS = 3;

    const totalRows = Math.ceil(formats.length / COLUMNS);
    const totalBlockHeight = (totalRows - 1) * ROW_SPACING;
    const startY = parentNode.y - (totalBlockHeight / 2);

    const newNodes: NodeData[] = [];
    
    formats.forEach((format, index) => {
      const row = Math.floor(index / COLUMNS);
      const col = index % COLUMNS;
      
      const newId = `creative-${Date.now()}-${index}`;
      const nodeData: NodeData = {
        id: newId, 
        type: NodeType.CREATIVE, 
        parentId: nodeId,
        title: format, 
        description: "Initializing Generation...", 
        format: format,
        isLoading: true, 
        x: parentNode.x + HORIZONTAL_GAP + (col * COL_SPACING), 
        y: startY + (row * ROW_SPACING),
        stage: CampaignStage.TESTING,
        meta: { 
            personaName: parentNode.meta?.personaName,
            angle: parentNode.title, // Critical: Pass angle to child for context
        }
      };
      newNodes.push(nodeData);
      addNode(nodeData);
      addEdge(nodeId, newId);
    });

    // GENERATION PROCESS
    for (const node of newNodes) {
        // Stagger execution slightly to prevent immediate rate limits, though we have retry logic
        if (newNodes.indexOf(node) > 0) await new Promise(resolve => setTimeout(resolve, 800));

        try {
            const personaName = parentNode.meta?.personaName || "User";
            // UPDATED: Pass full persona metadata for psychological context
            // parentNode.meta holds the persona object (name, profile, deepFear, etc) if available
            const personaMeta = parentNode.meta || { name: personaName };

            const angle = parentNode.title;
            const fmt = node.format as CreativeFormat;
            
            let accumulatedInput = 0;
            let accumulatedOutput = 0;
            let imageCount = 0;

            // 1. STRATEGIST AGENT (The Bridge) - ART DIRECTOR UPGRADE
            // Generate a cohesive concept first.
            updateNode(node.id, { description: "Art Director: Defining visual style..." });
            const conceptResult = await generateCreativeConcept(project, personaName, angle, fmt);
            accumulatedInput += conceptResult.inputTokens;
            accumulatedOutput += conceptResult.outputTokens;
            const concept = conceptResult.data;

            // 2. COPYWRITER AGENT (Uses Concept)
            updateNode(node.id, { description: "Copywriter: Drafting..." });
            // UPDATED: Calling with full persona object
            const copyResult = await generateAdCopy(project, personaMeta, concept);
            accumulatedInput += copyResult.inputTokens;
            accumulatedOutput += copyResult.outputTokens;
            const adCopy = copyResult.data;

            // 3. COMPLIANCE CHECK (Safety Layer)
            const complianceStatus = await checkAdCompliance(adCopy);
            adCopy.complianceNotes = complianceStatus;

            // 4. VISUALIZER AGENT (Uses Concept + Format + Style)
            updateNode(node.id, { description: "Visualizer: Rendering..." });
            
            const imgResult = await generateCreativeImage(
                project, 
                personaName, 
                angle, 
                fmt, 
                concept.visualScene, 
                concept.visualStyle, 
                concept.technicalPrompt,
                "1:1"
            );
            
            accumulatedInput += imgResult.inputTokens;
            accumulatedOutput += imgResult.outputTokens;
            const imageUrl = imgResult.data;
            if (imageUrl) imageCount++;

            // 5. CAROUSEL HANDLER (Optional)
            let carouselImages: string[] = [];
            const isCarousel = (
                fmt === CreativeFormat.CAROUSEL_EDUCATIONAL ||
                fmt === CreativeFormat.CAROUSEL_TESTIMONIAL ||
                fmt === CreativeFormat.CAROUSEL_PANORAMA ||
                fmt === CreativeFormat.CAROUSEL_PHOTO_DUMP ||
                fmt === CreativeFormat.CAROUSEL_REAL_STORY // NEW Check
            );
            
            if (isCarousel) {
                // Pass the concept's visual style to the carousel generator for consistency
                const slidesResult = await generateCarouselSlides(
                    project, 
                    fmt, 
                    angle, 
                    concept.visualScene, 
                    concept.visualStyle, 
                    concept.technicalPrompt
                );
                accumulatedInput += slidesResult.inputTokens;
                accumulatedOutput += slidesResult.outputTokens;
                carouselImages = slidesResult.data;
                imageCount += carouselImages.length;
            }

            // COST CALCULATION (Gemini 2.5 Flash Pricing)
            const inputCost = (accumulatedInput / 1000000) * 0.30;
            const outputCost = (accumulatedOutput / 1000000) * 2.50;
            const imgCost = imageCount * 0.039;
            const totalCost = inputCost + outputCost + imgCost;

            updateNode(node.id, { 
                isLoading: false, 
                description: adCopy.primaryText.slice(0, 100) + "...",
                imageUrl: imageUrl || undefined,
                carouselImages: carouselImages.length > 0 ? carouselImages : undefined,
                adCopy: adCopy,
                inputTokens: accumulatedInput,
                outputTokens: accumulatedOutput,
                estimatedCost: totalCost,
                meta: { 
                    ...node.meta, 
                    visualScene: concept.visualScene,
                    visualStyle: concept.visualStyle,
                    technicalPrompt: concept.technicalPrompt
                }, // Save style context for regeneration
                variableIsolated: concept.rationale // NEW: From Playbook
            });
        } catch (e) {
            console.error("Error generating creative node", e);
            updateNode(node.id, { isLoading: false, description: "Generation Failed" });
        }
    }
    updateNode(nodeId, { isLoading: false });
  };

  const handleNodeAction = async (action: string, nodeId: string) => {
    const parentNode = nodes.find(n => n.id === nodeId);
    if (!parentNode) return;

    if (action === 'expand_personas') {
      updateNode(nodeId, { isLoading: true });
      try {
          const result = await generatePersonas(project);
          const personas = result.data;
          
          const HORIZONTAL_GAP = 600;
          const VERTICAL_SPACING = 800;
          const totalHeight = (personas.length - 1) * VERTICAL_SPACING;
          const startY = parentNode.y - (totalHeight / 2);
          
          personas.forEach((p: any, index: number) => {
            const newNodeId = `persona-${Date.now()}-${index}`;
            addNode({
              id: newNodeId, type: NodeType.PERSONA, parentId: nodeId,
              title: p.name, 
              // UPDATED: Use profile and motivation for richer description
              description: `${p.profile || p.motivation}`,
              x: parentNode.x + HORIZONTAL_GAP, y: startY + (index * VERTICAL_SPACING),
              meta: p, stage: CampaignStage.TESTING,
              inputTokens: result.inputTokens / 3, // rough split
              outputTokens: result.outputTokens / 3,
              estimatedCost: ((result.inputTokens/1000000)*0.3 + (result.outputTokens/1000000)*2.5) / 3
            });
            addEdge(nodeId, newNodeId);
          });
      } catch (e) { alert("Quota exceeded."); }
      updateNode(nodeId, { isLoading: false });
    }

    if (action === 'expand_angles') {
      updateNode(nodeId, { isLoading: true });
      try {
          const pMeta = parentNode.meta || {};
          const result = await generateAngles(project, pMeta.name, pMeta.motivation);
          const angles = result.data;
          
          const HORIZONTAL_GAP = 550;
          const VERTICAL_SPACING = 350;
          const totalHeight = (angles.length - 1) * VERTICAL_SPACING;
          const startY = parentNode.y - (totalHeight / 2);

          angles.forEach((a: any, index: number) => {
            const newNodeId = `angle-${Date.now()}-${index}`;
            addNode({
              id: newNodeId, type: NodeType.ANGLE, parentId: nodeId,
              title: a.headline, description: `Hook: ${a.painPoint}`,
              x: parentNode.x + HORIZONTAL_GAP, y: startY + (index * VERTICAL_SPACING),
              meta: { ...a, personaName: pMeta.name }, stage: CampaignStage.TESTING,
              testingTier: a.testingTier, // NEW
              inputTokens: result.inputTokens / 3,
              outputTokens: result.outputTokens / 3,
              estimatedCost: ((result.inputTokens/1000000)*0.3 + (result.outputTokens/1000000)*2.5) / 3
            });
            addEdge(nodeId, newNodeId);
          });
      } catch (e) { alert("Quota exceeded."); }
      updateNode(nodeId, { isLoading: false });
    }

    if (action === 'generate_creatives') {
      setTargetAngleId(nodeId);
      setIsFormatModalOpen(true);
    }

    if (action === 'promote_creative') {
       // Deep Clone to move to Vault
       const newId = `${nodeId}-vault`;
       addNode({
           ...parentNode,
           id: newId,
           stage: CampaignStage.SCALING,
           x: 0, 
           y: 0,
           parentId: null // Vault items are root in their view
       });
       // Leave a ghost behind
       updateNode(nodeId, { isGhost: true });
       // Switch view automatically
       setActiveView('VAULT');
    }

    if (action === 'remix_creative') {
        updateNode(nodeId, { isLoading: true });
        // Simulation: Reset metrics to fresh
        setTimeout(() => {
            updateNode(nodeId, { 
                isLoading: false,
                metrics: { ...parentNode.metrics!, ctr: 2.5, cpa: 15, roas: 3.2, ageHours: 0 }, // Reset age
                isLosing: false,
                isWinning: true
            });
        }, 2000);
    }
  };

  // ANDROMEDA PLAYBOOK: Phased Analysis Simulation
  const runSimulation = () => {
    setSimulating(true);
    
    // Simulate metrics for all leaf nodes (Creatives) in LAB
    const creatives = nodes.filter(n => n.type === NodeType.CREATIVE && n.stage === CampaignStage.TESTING && !n.isGhost);
    
    creatives.forEach(node => {
        // Increment Age
        const currentAge = (node.metrics?.ageHours || 0) + 24; // Advance 1 day (24h)
        
        let phase = AnalysisPhase.PHASE_1;
        if (currentAge > 72 && currentAge <= 168) phase = AnalysisPhase.PHASE_2;
        if (currentAge > 168 && currentAge <= 336) phase = AnalysisPhase.PHASE_3;
        if (currentAge > 336) phase = AnalysisPhase.PHASE_4;

        // Base Performance (Random seed for the node)
        const basePerformance = Math.random(); // 0 to 1
        
        let spend = (node.metrics?.spend || 0);
        let cpa = 0;
        let roas = 0;
        let ctr = 0;
        let aiInsight = "";
        let isWinning = false;
        let isLosing = false;

        // Phase 1: Launch (0-72h) - HIGH VOLATILITY, NO ROAS
        if (phase === AnalysisPhase.PHASE_1) {
             spend += Math.floor(Math.random() * 50) + 20;
             ctr = parseFloat((Math.random() * 4).toFixed(2)); // Volatile CTR
             roas = 0; // Don't show ROAS yet
             cpa = 0; // Don't show CPA yet
             aiInsight = "PHASE 1 (Learning): Volatility detected. Do not touch. 72-hour rule active.";
        }
        
        // Phase 2: Health Check (Day 4-7) - CTR STABILIZES, CPM CHECK
        else if (phase === AnalysisPhase.PHASE_2) {
             spend += Math.floor(Math.random() * 100) + 50;
             ctr = parseFloat((basePerformance * 3 + 0.5).toFixed(2)); // Stabilizing
             // If CTR is bad (< 0.8%), kill it.
             if (ctr < 0.8) {
                 isLosing = true;
                 aiInsight = "PHASE 2 (Health): Thumbstop Rate < 1%. Kill immediately.";
             } else {
                 aiInsight = "PHASE 2 (Health): Thumbstop healthy. Monitoring conversion.";
             }
        }

        // Phase 3: Perf Eval (Day 8-14) - ROAS & CPA MATTERS
        else if (phase === AnalysisPhase.PHASE_3) {
             spend += Math.floor(Math.random() * 200) + 100;
             ctr = parseFloat((basePerformance * 3 + 0.5).toFixed(2));
             roas = parseFloat((basePerformance * 5).toFixed(2));
             cpa = Math.floor((1 / basePerformance) * 20) + 5;

             if (roas > 2.0) {
                 isWinning = true;
                 aiInsight = "PHASE 3 (Eval): Winner detected (ROAS > 2.0). Prepare to scale.";
             } else if (roas < 1.0) {
                 isLosing = true;
                 aiInsight = "PHASE 3 (Eval): Burner (ROAS < 1.0). Turn off.";
             } else {
                 aiInsight = "PHASE 3 (Eval): Mediocre. Iterate hook.";
             }
        }

        // Phase 4: Scaling (Day 15+)
        else {
             spend += Math.floor(Math.random() * 500) + 200;
             ctr = parseFloat((basePerformance * 3 + 0.5).toFixed(2));
             roas = parseFloat((basePerformance * 5).toFixed(2));
             cpa = Math.floor((1 / basePerformance) * 20) + 5;
             if (roas > 2.0) isWinning = true;
             aiInsight = "PHASE 4 (Scale): Horizontal scaling recommended.";
        }
        
        updateNode(node.id, {
            metrics: { 
                spend, 
                cpa, 
                roas, 
                impressions: spend * 40, 
                ctr,
                ageHours: currentAge
            },
            analysisPhase: phase,
            isWinning,
            isLosing,
            aiInsight
        });
    });

    setTimeout(() => setSimulating(false), 1500);
  };

  const handleSelectFormat = (fmt: CreativeFormat) => {
      const newSet = new Set(selectedFormats);
      if (newSet.has(fmt)) newSet.delete(fmt);
      else newSet.add(fmt);
      setSelectedFormats(newSet);
  };

  const confirmFormatSelection = () => {
      if (targetAngleId && selectedFormats.size > 0) {
          executeGeneration(targetAngleId, Array.from(selectedFormats));
          setIsFormatModalOpen(false);
          setSelectedFormats(new Set());
          setTargetAngleId(null);
      }
  };

  // Drag and Drop for Image Analysis
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && isConfigOpen) {
       // Trigger image analysis
       setIsAnalyzingImage(true);
       const reader = new FileReader();
       reader.onloadend = async () => {
          try {
             const context = await analyzeImageContext(reader.result as string);
             setProject(prev => ({
                 ...prev,
                 productName: context.productName,
                 productDescription: context.productDescription,
                 targetAudience: context.targetAudience,
                 targetCountry: context.targetCountry || prev.targetCountry
             }));
              // Update Root Node
              setNodes(prev => prev.map(n => n.type === NodeType.ROOT ? {
                  ...n,
                  title: context.productName,
                  description: context.productDescription
              } : n));
          } catch(e) { console.error(e); }
          setIsAnalyzingImage(false);
       };
       reader.readAsDataURL(file);
    }
  };

  return (
    <HashRouter>
    <div className="w-screen h-screen bg-slate-50 flex overflow-hidden text-slate-900" onDragOver={handleDragOver} onDrop={handleDrop}>
      
      {/* --- LEFT SIDEBAR --- */}
      <div className="w-16 bg-white border-r border-slate-200 flex flex-col items-center py-6 z-20 shadow-sm">
        <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl flex items-center justify-center mb-8 shadow-lg shadow-blue-500/20">
            <span className="text-white font-display font-bold text-xl">A</span>
        </div>
        
        <div className="flex flex-col gap-6 w-full">
            <button 
                onClick={() => setActiveView('LAB')}
                className={`w-full relative py-3 flex justify-center transition-all ${activeView === 'LAB' ? 'text-blue-600' : 'text-slate-400 hover:text-slate-600'}`}
            >
                <Microscope className="w-6 h-6" />
                {activeView === 'LAB' && <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-blue-600 rounded-l-full" />}
            </button>

            <button 
                onClick={() => setActiveView('VAULT')}
                className={`w-full relative py-3 flex justify-center transition-all ${activeView === 'VAULT' ? 'text-amber-500' : 'text-slate-400 hover:text-slate-600'}`}
            >
                <Package className="w-6 h-6" />
                {activeView === 'VAULT' && <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-amber-500 rounded-l-full" />}
            </button>
        </div>

        <div className="mt-auto flex flex-col gap-6 mb-4">
             <button onClick={() => setIsConfigOpen(true)} className="p-2 rounded-lg bg-slate-100 text-slate-500 hover:bg-slate-200 transition-colors">
                 <Settings className="w-5 h-5" />
             </button>
        </div>
      </div>

      {/* --- MAIN AREA --- */}
      <div className="flex-1 relative">
        <Canvas 
          ref={canvasRef}
          nodes={activeView === 'LAB' ? labNodes : vaultNodes}
          edges={activeView === 'LAB' ? labEdges : []}
          onNodeAction={handleNodeAction}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          onNodeMove={handleNodeMove}
        />

        {/* TOP BAR */}
        <div className="absolute top-0 left-0 w-full h-16 bg-white/80 backdrop-blur-sm border-b border-slate-200 flex items-center justify-between px-6 z-10">
            <div>
                <h1 className="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                    {activeView === 'LAB' ? <><Microscope className="w-4 h-4"/> Testing Lab</> : <><Package className="w-4 h-4 text-amber-500"/> Creative Vault</>}
                </h1>
                <p className="text-xs text-slate-400 font-mono">{activeView === 'LAB' ? `${labNodes.length} Assets Active` : `${vaultNodes.length} Winning Assets`}</p>
            </div>

            <div className="flex items-center gap-4">
                 {activeView === 'LAB' && (
                     <button 
                        onClick={runSimulation}
                        disabled={simulating}
                        className="px-4 py-2 bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:border-slate-300 text-xs font-bold rounded-lg shadow-sm transition-all flex items-center gap-2"
                     >
                        <Activity className={`w-4 h-4 ${simulating ? 'animate-spin text-blue-500' : 'text-emerald-500'}`} />
                        {simulating ? 'Simulating +24h...' : 'Run Daily Simulation'}
                     </button>
                 )}
            </div>
        </div>
      </div>

      {/* --- INSPECTOR --- */}
      {selectedNode && (
          <div className="w-[400px] h-full z-30 relative">
            <Inspector 
                node={selectedNode} 
                onClose={() => setSelectedNodeId(null)} 
                onUpdate={updateNode}
                onRegenerate={handleRegenerateNode}
                onPromote={(id) => handleNodeAction('promote_creative', id)} // Pass promoter
                project={project}
            />
          </div>
      )}

      {/* --- PROJECT CONFIG MODAL --- */}
      {isConfigOpen && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white w-full max-w-4xl rounded-2xl shadow-2xl flex overflow-hidden max-h-[90vh]">
             
             {/* CONFIG LEFT: FIRECRAWL & IMAGE IMPORT */}
             <div className="w-1/3 bg-slate-50 p-8 border-r border-slate-200 overflow-y-auto">
                 <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">Import Context</h3>
                 
                 <div className="space-y-6">
                    {/* URL INPUT */}
                    <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                        <label className="text-xs font-bold text-slate-700 mb-2 block flex items-center gap-2"><Globe className="w-3.5 h-3.5" /> Landing Page URL</label>
                        <div className="flex gap-2">
                            <input 
                                type="text" 
                                className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs" 
                                placeholder="https://..."
                                value={landingPageUrl}
                                onChange={(e) => setLandingPageUrl(e.target.value)}
                            />
                        </div>
                        <button 
                            onClick={handleAnalyzeUrl}
                            disabled={isAnalyzing}
                            className="mt-2 w-full py-2 bg-blue-50 hover:bg-blue-100 text-blue-600 text-xs font-bold rounded-lg transition-colors"
                        >
                            {isAnalyzing ? "Scanning..." : "Analyze Site"}
                        </button>
                    </div>

                    <div className="flex items-center gap-4">
                        <div className="h-px bg-slate-200 flex-1"></div>
                        <span className="text-[10px] text-slate-400 font-bold">OR</span>
                        <div className="h-px bg-slate-200 flex-1"></div>
                    </div>

                    {/* IMAGE UPLOAD */}
                    <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm relative group">
                        <label className="text-xs font-bold text-slate-700 mb-2 block flex items-center gap-2"><ImageIcon className="w-3.5 h-3.5" /> Product Image Analysis</label>
                        <div 
                            className="w-full h-32 border-2 border-dashed border-slate-300 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:bg-slate-50 hover:border-blue-400 transition-colors"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            {isAnalyzingImage ? (
                                <div className="flex flex-col items-center gap-2">
                                    <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                                    <span className="text-xs text-blue-600 font-medium">Analyzing...</span>
                                </div>
                            ) : (
                                <>
                                    <Upload className="w-6 h-6 text-slate-400 mb-2" />
                                    <span className="text-xs text-slate-500">Drop or Click to Upload</span>
                                </>
                            )}
                        </div>
                        <input 
                            type="file" 
                            ref={fileInputRef} 
                            className="hidden" 
                            accept="image/*" 
                            onChange={handleImageUpload}
                        />
                    </div>
                 </div>
             </div>

             {/* CONFIG RIGHT: MANUAL EDIT & STRATEGY */}
             <div className="w-2/3 p-8 overflow-y-auto">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h2 className="text-2xl font-display font-bold text-slate-900">Project Brief</h2>
                        <p className="text-sm text-slate-500">Define the core strategy. AI will adhere to this.</p>
                    </div>
                    <button onClick={() => setIsConfigOpen(false)} className="p-2 hover:bg-slate-100 rounded-full"><X className="w-5 h-5 text-slate-400" /></button>
                </div>

                <div className="grid grid-cols-2 gap-6 mb-6">
                    <div className="col-span-2">
                        <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Product Name</label>
                        <input className="w-full text-lg font-bold text-slate-900 border-b-2 border-slate-100 focus:border-blue-500 outline-none py-2 transition-colors" value={project.productName} onChange={e => setProject({...project, productName: e.target.value})} />
                    </div>
                    <div className="col-span-2">
                        <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Value Proposition (Description)</label>
                        <textarea className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-100 outline-none" rows={2} value={project.productDescription} onChange={e => setProject({...project, productDescription: e.target.value})} />
                    </div>
                    
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Target Audience</label>
                        <input className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm" value={project.targetAudience} onChange={e => setProject({...project, targetAudience: e.target.value})} />
                    </div>
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Target Country</label>
                        <div className="relative">
                            <Globe className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
                            <input className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-9 pr-3 py-2 text-sm" value={project.targetCountry || ''} onChange={e => setProject({...project, targetCountry: e.target.value})} placeholder="e.g. Indonesia" />
                        </div>
                    </div>
                </div>

                <div className="h-px bg-slate-100 my-8"></div>

                <h3 className="text-sm font-bold text-slate-900 uppercase tracking-widest mb-4 flex items-center gap-2"><Target className="w-4 h-4 text-pink-500"/> Strategic Direction</h3>
                
                <div className="grid grid-cols-2 gap-6">
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Brand Voice</label>
                        <input className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm" value={project.brandVoice || ''} onChange={e => setProject({...project, brandVoice: e.target.value})} placeholder="e.g. Witty, Gen-Z, Professional" />
                    </div>
                    <div>
                        <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">The Offer</label>
                        <input className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-sm" value={project.offer || ''} onChange={e => setProject({...project, offer: e.target.value})} placeholder="e.g. 50% Off, Buy 1 Get 1" />
                    </div>

                    <div className="col-span-2 grid grid-cols-2 gap-6 p-4 bg-slate-50 rounded-xl border border-slate-200">
                        {/* FUNNEL STAGE REMOVED - AUTO SYNCED */}
                        
                        <div>
                            <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Market Awareness</label>
                            <select 
                                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm cursor-pointer hover:border-blue-300 transition-colors"
                                value={project.marketAwareness}
                                onChange={(e) => handleAwarenessChange(e.target.value as MarketAwareness)}
                            >
                                {Object.values(MarketAwareness).map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                            <p className="text-[10px] text-slate-400 mt-1.5 leading-tight">
                                Determines strategy. <br/>
                                <b>Unaware/Problem</b> = Top of Funnel.<br/>
                                <b>Solution</b> = Middle of Funnel.<br/>
                                <b>Product/Most</b> = Bottom of Funnel.
                            </p>
                        </div>
                        <div>
                             <label className="text-xs font-bold text-slate-500 uppercase mb-1 block">Copy Framework</label>
                             <select 
                                className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm cursor-pointer hover:border-blue-300 transition-colors"
                                value={project.copyFramework}
                                onChange={(e) => setProject({...project, copyFramework: e.target.value as CopyFramework})}
                            >
                                {Object.values(CopyFramework).map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                        </div>
                    </div>
                </div>

                <div className="h-px bg-slate-100 my-8"></div>

                <div className="mb-8">
                     <label className="text-xs font-bold text-slate-500 uppercase mb-2 block flex items-center gap-2"><ImageIcon className="w-3.5 h-3.5" /> Product Reference Image (Optional)</label>
                     <div className="flex items-center gap-4">
                         <div 
                            className="w-24 h-24 bg-slate-50 border-2 border-dashed border-slate-200 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 transition-colors relative overflow-hidden"
                            onClick={() => productRefInputRef.current?.click()}
                         >
                            {project.productReferenceImage ? (
                                <img src={project.productReferenceImage} className="w-full h-full object-cover" />
                            ) : <Upload className="w-6 h-6 text-slate-300" />}
                         </div>
                         <div className="flex-1">
                             <p className="text-xs text-slate-500 leading-relaxed">Upload a clear photo of your product. The AI will try to include this product in the generated visuals.</p>
                         </div>
                     </div>
                     <input type="file" ref={productRefInputRef} className="hidden" accept="image/*" onChange={handleProductRefUpload}/>
                </div>

                <button 
                    onClick={() => setIsConfigOpen(false)}
                    className="w-full py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-blue-500/30 transition-all transform hover:scale-[1.01]"
                >
                    Save Strategy & Enter Lab
                </button>
             </div>
          </div>
        </div>
      )}

      {/* --- FORMAT SELECTION MODAL --- */}
      {isFormatModalOpen && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-8">
              <div className="bg-white w-full max-w-5xl h-[80vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">
                  <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                      <div>
                          <h2 className="text-xl font-display font-bold text-slate-900">Select Creative Formats</h2>
                          <p className="text-sm text-slate-500">Choose formats to generate for this angle.</p>
                      </div>
                      <div className="flex gap-3">
                          <button onClick={() => setIsFormatModalOpen(false)} className="px-4 py-2 text-slate-500 hover:bg-slate-200 rounded-lg font-bold text-sm">Cancel</button>
                          <button 
                            onClick={confirmFormatSelection} 
                            disabled={selectedFormats.size === 0}
                            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-lg font-bold text-sm shadow-lg shadow-blue-500/20 transition-all"
                          >
                              Generate {selectedFormats.size} Creatives
                          </button>
                      </div>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-8 bg-slate-50/50">
                      <div className="grid grid-cols-2 gap-8">
                          {Object.entries(FORMAT_GROUPS).map(([group, formats]) => (
                              <div key={group} className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                                  <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                                      {group.includes("Instagram") ? <Smartphone className="w-3.5 h-3.5"/> : <Layers className="w-3.5 h-3.5"/>}
                                      {group}
                                  </h3>
                                  <div className="grid grid-cols-2 gap-3">
                                      {formats.map(fmt => (
                                          <button
                                              key={fmt}
                                              onClick={() => handleSelectFormat(fmt)}
                                              className={`text-left px-3 py-2.5 rounded-lg text-xs font-medium border transition-all ${selectedFormats.has(fmt) ? 'bg-blue-50 border-blue-500 text-blue-700 shadow-sm' : 'bg-slate-50 border-transparent hover:bg-slate-100 text-slate-600'}`}
                                          >
                                              {fmt}
                                          </button>
                                      ))}
                                  </div>
                              </div>
                          ))}
                      </div>
                  </div>
              </div>
          </div>
      )}

    </div>
    </HashRouter>
  );
};

export default App;