
// Lightweight wrapper for Firecrawl API
// We use direct fetch here to avoid installing node-specific dependencies in the browser environment

const API_KEY = "fc-787aa473762c40b092cf922840e528a6";
const API_URL = "https://api.firecrawl.dev/v1/scrape";

export interface ScrapeResult {
  success: boolean;
  markdown?: string;
  error?: string;
}

export const scrapeLandingPage = async (url: string): Promise<ScrapeResult> => {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        url: url,
        formats: ["markdown"]
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `Firecrawl Error: ${response.status}`);
    }

    const data = await response.json();
    
    // Firecrawl response structure: { success: true, data: { markdown: "..." } }
    if (data.success && data.data && data.data.markdown) {
      return {
        success: true,
        markdown: data.data.markdown
      };
    } else {
        return {
            success: false,
            error: "No markdown returned from Firecrawl"
        }
    }

  } catch (error: any) {
    console.error("Firecrawl Scrape Failed:", error);
    return {
      success: false,
      error: error.message || "Unknown error during scraping"
    };
  }
};
