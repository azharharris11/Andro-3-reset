import { GoogleGenAI, Type, Modality } from "@google/genai";
import { ProjectContext, CreativeFormat, AdCopy, CreativeConcept, GenResult } from "../types";

// Initialize the client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- UTILS ---

function extractJSON<T>(text: string): T {
  try {
    const cleanText = text.replace(/```json/g, '').replace(/```/g, '').trim();
    return JSON.parse(cleanText) as T;
  } catch (e) {
    console.error("JSON Parse Error", e, text);
    return {} as T;
  }
}

const VISUAL_STYLES = [
  "Shot on 35mm film, Fujifilm Pro 400H, grainy texture, nostalgic",
  "High-end studio photography, softbox lighting, sharp focus, 8k resolution",
  "Gen-Z aesthetic, flash photography, direct flash, high contrast, candid",
  "Cinematic lighting, golden hour, shallow depth of field, bokeh background",
  "Clean minimalist product photography, bright airy lighting, pastel tones"
];

const getRandomStyle = () => VISUAL_STYLES[Math.floor(Math.random() * VISUAL_STYLES.length)];

// --- ANALYSIS FUNCTIONS ---

export const analyzeLandingPageContext = async (markdown: string): Promise<ProjectContext> => {
  const model = "gemini-2.5-flash";
  
  // SYSTEM: Gather Data (Step 1 of the Tweet)
  const response = await ai.models.generateContent({
    model,
    contents: `You are a Data Analyst for a Direct Response Agency. 
    Analyze the following raw data (Landing Page Content) to extract the foundational truths.
    
    RAW DATA:
    ${markdown.substring(0, 30000)}
    `,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          productName: { type: Type.STRING },
          productDescription: { type: Type.STRING, description: "A punchy, benefit-driven 1-sentence value prop." },
          targetAudience: { type: Type.STRING, description: "Specific demographics and psychographics." },
          targetCountry: { type: Type.STRING },
          brandVoice: { type: Type.STRING },
          offer: { type: Type.STRING, description: "The primary hook or deal found on the page." }
        },
        required: ["productName", "productDescription", "targetAudience"]
      }
    }
  });

  const data = extractJSON<Partial<ProjectContext>>(response.text || "{}");
  
  return {
    productName: data.productName || "Unknown Product",
    productDescription: data.productDescription || "",
    targetAudience: data.targetAudience || "General Audience",
    targetCountry: data.targetCountry || "USA",
    brandVoice: data.brandVoice || "Professional",
    offer: data.offer || "Shop Now",
    landingPageUrl: "" 
  } as ProjectContext;
};

export const analyzeImageContext = async (base64Image: string): Promise<ProjectContext> => {
  const base64Data = base64Image.split(',')[1] || base64Image;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: {
      parts: [
        { inlineData: { mimeType: "image/jpeg", data: base64Data } },
        { text: "Analyze this product image. Extract the Product Name (if visible, otherwise guess), a compelling Description, and the likely Target Audience." }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          productName: { type: Type.STRING },
          productDescription: { type: Type.STRING },
          targetAudience: { type: Type.STRING },
          targetCountry: { type: Type.STRING }
        },
        required: ["productName", "productDescription"]
      }
    }
  });

  const data = extractJSON<Partial<ProjectContext>>(response.text || "{}");

  return {
    productName: data.productName || "Analyzed Product",
    productDescription: data.productDescription || "A revolutionary product.",
    targetAudience: data.targetAudience || "General Audience",
    targetCountry: "USA", 
    brandVoice: "Visual & Aesthetic",
    offer: "Check it out"
  } as ProjectContext;
};

// --- GENERATION FUNCTIONS ---

export const generatePersonas = async (project: ProjectContext): Promise<GenResult<any[]>> => {
  const model = "gemini-2.5-flash";
  
  // SYSTEM: Identity Mapping (Step 3: "Tag with Psychology" - Identity)
  const prompt = `
    You are a Consumer Psychologist specializing in ${project.targetCountry || "the target market"}.
    
    PRODUCT CONTEXT:
    Product: ${project.productName}
    Details: ${project.productDescription}
    
    TASK:
    Define 3 distinct "Avatars" based on their IDENTITY and DEEP PSYCHOLOGICAL NEEDS.
    Do not just list demographics. List who they *are* vs who they *want to be* (The Gap).

    We are looking for:
    1. The Skeptic / Logic Buyer (Identity: "I am smart, I research, I don't get fooled.")
    2. The Status / Aspirer (Identity: "I want to be admired/successful/beautiful.")
    3. The Anxious / Urgent Solver (Identity: "I need safety/certainty/speed.")

    *Cultural nuance mandatory for ${project.targetCountry}. If Indonesia, mention specific local behaviors (e.g., 'Kaum Mendang-Mending', 'Social Climber').*
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            name: { type: Type.STRING },
            profile: { type: Type.STRING, description: "Demographics + Identity Statement" },
            motivation: { type: Type.STRING, description: "The 'Gap' between current self and desired self." },
            deepFear: { type: Type.STRING, description: "What are they afraid of losing?" },
          },
          required: ["name", "profile", "motivation"]
        }
      }
    }
  });

  return {
    data: extractJSON(response.text || "[]"),
    inputTokens: response.usageMetadata?.promptTokenCount || 0,
    outputTokens: response.usageMetadata?.candidatesTokenCount || 0
  };
};

export const generateAngles = async (project: ProjectContext, personaName: string, personaMotivation: string): Promise<GenResult<any[]>> => {
  const model = "gemini-2.5-flash";

  // SYSTEM: Andromeda Strategy (Tier Selection & Prioritization)
  const prompt = `
    You are a Direct Response Strategist applying the "Andromeda Testing Playbook".
    
    CONTEXT:
    Product: ${project.productName}
    Persona: ${personaName}
    Deep Motivation: ${personaMotivation}
    Target Country: ${project.targetCountry}
    
    TASK:
    1. "Gather Data": Brainstorm 10 raw angles/hooks.
    2. "Prioritize": Rank by Market Size, Urgency, Differentiation.
    3. "Assign Tier": Assign a Testing Tier to each angle based on its nature:
       - TIER 1 (Concept Isolation): Big, bold, new ideas. High risk/reward.
       - TIER 2 (Persona Isolation): Specifically tailored to this persona's fear/desire.
       - TIER 3 (Sprint Isolation): A simple iteration or direct offer.
    
    OUTPUT:
    Return ONLY the Top 3 High-Potential Insights.
    
    *For ${project.targetCountry}: Ensure the angles fit the local culture.*
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            headline: { type: Type.STRING, description: "The core Hook/Angle name" },
            painPoint: { type: Type.STRING, description: "The specific problem or insight" },
            psychologicalTrigger: { type: Type.STRING, description: "The principle used (e.g. Loss Aversion)" },
            testingTier: { type: Type.STRING, description: "TIER 1, TIER 2, or TIER 3" },
            hook: { type: Type.STRING, description: "The opening line or concept" }
          },
          required: ["headline", "painPoint", "psychologicalTrigger", "testingTier"]
        }
      }
    }
  });

  return {
    data: extractJSON(response.text || "[]"),
    inputTokens: response.usageMetadata?.promptTokenCount || 0,
    outputTokens: response.usageMetadata?.candidatesTokenCount || 0
  };
};

export const generateCreativeConcept = async (
  project: ProjectContext, 
  personaName: string, 
  angle: string, 
  format: CreativeFormat
): Promise<GenResult<CreativeConcept>> => {
  const model = "gemini-2.5-flash";

  // SYSTEM: Congruency Engine (The Jeans Rule)
  const awareness = project.marketAwareness || "Problem Aware";
  
  let awarenessInstruction = "";
  if (awareness.includes("Unaware") || awareness.includes("Problem")) {
      awarenessInstruction = `AWARENESS: LOW. Focus on SYMPTOM. Use Pattern Interrupt.`;
  } else if (awareness.includes("Solution")) {
      awarenessInstruction = `AWARENESS: MEDIUM. Focus on MECHANISM and SOCIAL PROOF.`;
  } else {
      awarenessInstruction = `AWARENESS: HIGH. Focus on URGENCY and OFFER.`;
  }

  const prompt = `
    # Role: Creative Director (Focus: Message & Imagery Congruency)

    **THE GOLDEN RULE OF CONGRUENCE:**
    Ads fail when the image matches the *product* but ignores the *message*.
    
    *   **Bad Example:** Headline: "Stretchiest Jeans ever" -> Image: Sexy model standing still (Fail).
    *   **Good Example:** Headline: "Stretchiest Jeans ever" -> Image: Close up of fabric stretching 2x wide (Pass).
    *   **Bad Example:** Headline: "Hubby thinks I'm hot" -> Image: Fabric close up (Fail).
    *   **Good Example:** Headline: "Hubby thinks I'm hot" -> Image: Husband looking at wife with jaw drop (Pass).

    **INPUTS:**
    Product: ${project.productName}
    Winning Insight (The Message): ${angle}
    Persona: ${personaName}
    Format: ${format}
    Context: ${project.targetCountry}
    ${awarenessInstruction}
    
    **TASK:**
    Create a concept where the VISUAL **proves** the HEADLINE.
    
    **OUTPUT REQUIREMENTS (JSON):**

    **1. Congruence Rationale:**
    Explain WHY this image matches this specific headline. "The headline promises X, so the image shows X happening."

    **2. TECHNICAL PROMPT (technicalPrompt):**
    A STRICT prompt for the Image Generator. 
    *   If format is text-heavy (e.g. Twitter, Notes), describe the BACKGROUND VIBE and UI details.
    *   If format is visual (e.g. Photography), the SUBJECT ACTION must match the HOOK.
    *   If format is "Benefit Pointers", describe the product/subject clearly so pointers can be added.
    *   If format is "Real Story", describe a candid UGC moment.

    **3. SCRIPT DIRECTION (copyAngle):**
    Instructions for the copywriter.
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          visualScene: { type: Type.STRING, description: "Director's Note" },
          visualStyle: { type: Type.STRING, description: "Aesthetic vibe" },
          technicalPrompt: { type: Type.STRING, description: "Strict prompt for Image Gen" },
          copyAngle: { type: Type.STRING, description: "Strategy for the copywriter" },
          rationale: { type: Type.STRING, description: "Strategic Hypothesis" },
          congruenceRationale: { type: Type.STRING, description: "Why the Image proves the Text (The Jeans Rule)" },
          hookComponent: { type: Type.STRING, description: "The Visual Hook element" },
          bodyComponent: { type: Type.STRING, description: "The Core Argument element" },
          ctaComponent: { type: Type.STRING, description: "The Call to Action element" }
        },
        required: ["visualScene", "visualStyle", "technicalPrompt", "copyAngle", "rationale", "congruenceRationale"]
      }
    }
  });

  return {
    data: extractJSON(response.text || "{}"),
    inputTokens: response.usageMetadata?.promptTokenCount || 0,
    outputTokens: response.usageMetadata?.candidatesTokenCount || 0
  };
};

export const generateAdCopy = async (
  project: ProjectContext, 
  persona: any, 
  concept: CreativeConcept
): Promise<GenResult<AdCopy>> => {
  const model = "gemini-2.5-flash";

  // SYSTEM: Ship Fast (Step 5)
  const isIndo = project.targetCountry?.toLowerCase().includes("indonesia");
  const languageInstruction = isIndo
    ? "Write in Bahasa Indonesia. Use 'Bahasa Marketing' (mix of persuasive & conversational). Use local power words (e.g., 'Slot Terbatas', 'Best Seller', 'Gak Nyesel')."
    : "Write in English (or native language). Use persuasive Direct Response copy.";

  // SYSTEM: HEADLINE CONTEXT LIBRARY (THE STATIC AD RULES)
  const prompt = `
    # Role: Senior Direct Response Copywriter (Static Ad Specialist)

    **MANDATORY INSTRUCTION:**
    ${languageInstruction}

    **THE HEADLINE CONTEXT LIBRARY (RULES):**
    1.  **Assume No One Knows You:** Treat the audience as COLD. Do not be vague. "I feel new" (BAD) vs "Bye-Bye Bloating" (GOOD).
    2.  **Clear > Clever:** Clarity drives conversions. No puns. No jargon. If they have to think, you lose.
    3.  **The "So That" Test (Transformation > Feature):** 
        *   Feature: "1000mAh Battery" (Boring).
        *   Transformation: "Listen to music for 48 hours straight" (Winner).
        *   *Rule:* Sell the AFTER state.
    4.  **Call Out the Audience/Pain:** Flag down the user immediately.
        *   "For Busy Moms..."
        *   "Knee Pain keeping you up?"
        *   "The last backpack a Digital Nomad will need."
    5.  **Scannability:** Under 7 words. High contrast thought.
    6.  **Visual Hierarchy:** The headline MUST match the image scene described below.

    **STRATEGY CONTEXT:**
    Product: ${project.productName}
    Offer: ${project.offer}
    Target: ${persona.name}
    
    **CONGRUENCE CONTEXT (IMAGE SCENE):**
    Visual Scene: "${concept.visualScene}"
    Rationale: "${concept.congruenceRationale}"
    
    **TASK:**
    Write the ad copy applying the rules above.

    **OUTPUT:**
    1. Primary Text: The main caption. Match the tone to the identity of the persona.
    2. Headline: Apply the "So That" test. Max 7 words. MUST be congruent with the Visual Scene.
    3. CTA: Clear instruction.
  `;

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          primaryText: { type: Type.STRING },
          headline: { type: Type.STRING },
          cta: { type: Type.STRING }
        },
        required: ["primaryText", "headline", "cta"]
      }
    }
  });

  return {
    data: extractJSON(response.text || "{}"),
    inputTokens: response.usageMetadata?.promptTokenCount || 0,
    outputTokens: response.usageMetadata?.candidatesTokenCount || 0
  };
};

export const checkAdCompliance = async (adCopy: AdCopy): Promise<string> => {
  const model = "gemini-2.5-flash";
  const response = await ai.models.generateContent({
    model,
    contents: `Check this Ad Copy for policy violations. If safe, return "SAFE".\nHeadline: ${adCopy.headline}\nText: ${adCopy.primaryText}`,
  });
  return response.text || "SAFE";
};

// --- IMAGE GENERATION (NANO BANANA) ---

export const generateCreativeImage = async (
  project: ProjectContext,
  personaName: string,
  angle: string,
  format: CreativeFormat,
  visualScene: string,
  visualStyle: string,
  technicalPrompt: string,
  aspectRatio: string = "1:1"
): Promise<GenResult<string | null>> => {
  
  const model = "gemini-2.5-flash-image";

  // 1. CULTURAL & SERVICE CONTEXT INJECTION
  const isIndo = project.targetCountry?.toLowerCase().includes("indonesia");
  const lowerDesc = project.productDescription.toLowerCase();
  
  // Is this a Service Business? (Used for fallback logic only)
  const isService = lowerDesc.includes("studio") || lowerDesc.includes("service") || lowerDesc.includes("jasa") || lowerDesc.includes("photography") || lowerDesc.includes("clinic");
  
  let culturePrompt = "";
  if (isIndo) {
     culturePrompt = " Indonesian aesthetic, Asian features, localized environment.";
     
     // Specific Case: Graduation/Wisuda
     if (lowerDesc.includes("graduation") || lowerDesc.includes("wisuda")) {
         culturePrompt += " Young Indonesian university student wearing a black graduation toga with university sash/selempang (Indonesian style), holding a tube or bouquet. Authentic Indonesian look (hijab optional but common).";
     }
  }

  // 2. PHOTOGRAPHY ENHANCERS
  const visualEnhancers = "Photorealistic, 8k resolution, highly detailed, shot on 35mm lens, depth of field, natural lighting.";

  // 3. STRICT FORMAT ENGINEERING
  // The logic here determines priority: Format > Service > Generic
  let finalPrompt = "";
  
  // CONGRUENCY INJECTION
  const contextInjection = `(Context: The image must match this headline: "${angle}").`;

  // === HIGH PRIORITY: DIRECT RESPONSE UI FORMATS ===
  if (format === CreativeFormat.TWITTER_REPOST || format === CreativeFormat.HANDHELD_TWEET) {
    finalPrompt = `A photorealistic close-up POV shot of a hand holding a modern smartphone. On the screen, a social media post is clearly visible. The text reads: "${angle}". The background is blurred ${visualScene}. High resolution, screen reflection, authentic UI.`;
  } 
  else if (format === CreativeFormat.PHONE_NOTES) {
    finalPrompt = `A close-up screenshot of the Apple Notes app UI on an iPhone. The text typed in the note is: "${angle}". The background is the standard paper texture of the Notes app. At the bottom, a checklist related to ${project.productName}. Photorealistic screen capture.`;
  }
  else if (format === CreativeFormat.STICKY_NOTE_REALISM) {
    finalPrompt = `A real yellow post-it sticky note stuck on a surface. Handwritten black marker text on the note says: "${angle}". Sharp focus on the text, realistic paper texture, soft shadows.`;
  }
  else if (format === CreativeFormat.SOCIAL_COMMENT_STACK) {
    finalPrompt = `A graphic design showing floating social media comment bubbles on a clean background. Top comment: "Highly recommended!". Middle comment: "${angle}". Bottom comment: "Booking mine now!". Clean UI, sharp text, modern aesthetic.`;
  }
  else if (format === CreativeFormat.BENEFIT_POINTERS) {
    // Force product/subject visualization with lines
    finalPrompt = `A high-quality product photography shot of ${project.productName} (or the main subject of the service). Clean background. There are sleek, modern graphic lines pointing to 3 key features of the subject. The style is "Anatomy Breakdown". ${visualEnhancers} ${culturePrompt}. (Note: The pointers are the main visual hook).`;
  }
  else if (format === CreativeFormat.US_VS_THEM) {
    finalPrompt = `A split screen comparison image. Left side (Them): Cloudy, sad, messy, labeled "Them". Right side (Us): Bright, happy, organized, labeled "Us". The subject is related to: ${project.productName}. ${visualEnhancers} ${culturePrompt}.`;
  }
  // === CAROUSEL FORMATS (Relies on specific technicalPrompt passed from generateCarouselSlides) ===
  else if (
      format === CreativeFormat.CAROUSEL_REAL_STORY || 
      format === CreativeFormat.CAROUSEL_EDUCATIONAL || 
      format === CreativeFormat.CAROUSEL_TESTIMONIAL
  ) {
      // Trust the technicalPrompt fully for Carousels as it contains Slide-specific info
      finalPrompt = `${technicalPrompt}. ${visualEnhancers} ${culturePrompt}`;
  }
  // === FALLBACK: SERVICE BUSINESS LOGIC ===
  // Only apply "Service Logic" if it's NOT one of the specific formats above
  else if (isService) {
      // For services, we focus on the RESULT or the EXPERIENCE, not a "Product Box"
      finalPrompt = `${contextInjection} ${technicalPrompt}. ${culturePrompt} ${visualEnhancers}. (Note: This is a service, do not show a retail box. Show the person experiencing the result).`;
  }
  // === FALLBACK: STANDARD GENERIC ===
  else {
     if (technicalPrompt && technicalPrompt.length > 20) {
         finalPrompt = `${contextInjection} ${technicalPrompt}. ${visualEnhancers} ${culturePrompt}`;
     } else {
         finalPrompt = `${contextInjection} ${visualScene}. Style: ${visualStyle || getRandomStyle()}. ${visualEnhancers} ${culturePrompt}`;
     }
  }

  const parts: any[] = [{ text: finalPrompt }];
  
  if (project.productReferenceImage) {
      const base64Data = project.productReferenceImage.split(',')[1] || project.productReferenceImage;
      parts.unshift({
          inlineData: { mimeType: "image/png", data: base64Data }
      });
      parts.push({ text: "Use the product/subject in the provided image as the reference. Maintain brand colors and visual identity." });
  }

  try {
    const response = await ai.models.generateContent({
      model,
      contents: { parts },
      config: { imageConfig: { aspectRatio: aspectRatio === "1:1" ? "1:1" : "9:16" } }
    });

    let imageUrl: string | null = null;
    if (response.candidates && response.candidates[0].content.parts) {
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
                imageUrl = `data:image/png;base64,${part.inlineData.data}`;
                break;
            }
        }
    }
    return {
      data: imageUrl,
      inputTokens: response.usageMetadata?.promptTokenCount || 0,
      outputTokens: response.usageMetadata?.candidatesTokenCount || 0
    };
  } catch (error) {
    console.error("Image Gen Error", error);
    return { data: null, inputTokens: 0, outputTokens: 0 };
  }
};

export const generateCarouselSlides = async (
  project: ProjectContext,
  format: CreativeFormat,
  angle: string,
  visualScene: string,
  visualStyle: string,
  technicalPrompt: string
): Promise<GenResult<string[]>> => {
    const slides: string[] = [];
    let totalInput = 0;
    let totalOutput = 0;

    // DYNAMIC STORYBOARDING FOR CAROUSELS
    // We break the single technicalPrompt into a 3-act structure
    
    for (let i = 1; i <= 3; i++) {
        let slidePrompt = "";
        
        if (format === CreativeFormat.CAROUSEL_REAL_STORY) {
             if (i === 1) slidePrompt = `Slide 1 (The Hook): A candid, slightly imperfect UGC-style photo showing the PROBLEM or PAIN POINT. The subject looks frustrated or tired. Context: ${angle}. Style: Handheld camera.`;
             if (i === 2) slidePrompt = `Slide 2 (The Turn): The subject discovers ${project.productName}. A close up shot of the product/service in use. Natural lighting.`;
             if (i === 3) slidePrompt = `Slide 3 (The Result): The subject looks relieved and happy. A glowing transformation result. Text overlay implied: "Saved my life".`;
        } 
        else if (format === CreativeFormat.CAROUSEL_EDUCATIONAL) {
             if (i === 1) slidePrompt = `Slide 1 (Title Card): Minimalist background with plenty of negative space for text. Visual icon representing the topic: ${angle}.`;
             if (i === 2) slidePrompt = `Slide 2 (The Method): A diagram or clear photo demonstrating the 'How To' aspect of the solution.`;
             if (i === 3) slidePrompt = `Slide 3 (Summary): A checklist visual or a final result shot showing success.`;
        }
        else {
            // Default Carousel Structure
            if (i === 1) slidePrompt = `${technicalPrompt}. Slide 1: The Hook/Problem. High tension visual.`;
            if (i === 2) slidePrompt = `${technicalPrompt}. Slide 2: The Solution/Process. Detailed macro shot.`;
            if (i === 3) slidePrompt = `${technicalPrompt}. Slide 3: The Result/CTA. Happy resolution.`;
        }

        const result = await generateCreativeImage(
            project, "User", angle, format, visualScene, visualStyle, slidePrompt, "1:1"
        );
        if (result.data) slides.push(result.data);
        totalInput += result.inputTokens;
        totalOutput += result.outputTokens;
    }

    return { data: slides, inputTokens: totalInput, outputTokens: totalOutput };
};

// --- AUDIO GENERATION ---

export const generateAdScript = async (project: ProjectContext, personaName: string, angle: string): Promise<string> => {
    const model = "gemini-2.5-flash";
    const lang = project.targetCountry?.toLowerCase().includes("indonesia") ? "Bahasa Indonesia (Colloquial/Gaul)" : "English";

    const response = await ai.models.generateContent({
        model,
        contents: `Write a 15-second TikTok/Reels UGC script for: ${project.productName}. Language: ${lang}. Angle: ${angle}. Keep it under 40 words. Hook the viewer instantly.`
    });
    return response.text || "Script generation failed.";
};

export const generateVoiceover = async (script: string, personaName: string): Promise<string | null> => {
    const spokenText = script.replace(/\[.*?\]/g, '').trim();
    let voiceName = 'Zephyr'; 
    if (personaName.toLowerCase().includes('skeptic') || personaName.toLowerCase().includes('man')) voiceName = 'Fenrir';
    if (personaName.toLowerCase().includes('status') || personaName.toLowerCase().includes('woman')) voiceName = 'Kore';

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: { parts: [{ text: spokenText }] },
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName } } }
            }
        });
        return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data || null;
    } catch (e) {
        console.error("TTS Error", e);
        return null;
    }
};
