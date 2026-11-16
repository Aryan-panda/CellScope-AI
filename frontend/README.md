# CellScope AI — Frontend Client

React 18, Vite, TypeScript, and Tailwind CSS web application for biomedical cell analysis.

## Features & Components

- **`src/components/Sidebar.tsx`**: Preset selectors, file drag-and-drop, parameter sliders (Confidence $\tau$, Min Area, Watershed Peak Distance, Opacity), and hardware status.
- **`src/components/KPICards.tsx`**: 4 hero metric cards (Total Nuclei, Average Area, Total Foreground, Cell Coverage) and watershed-split cluster indicators.
- **`src/components/TabNavigation.tsx`**: 5 core analysis workspaces:
  1. *Overview*: Triad view of microscopy view, segmented labels, and alpha blend.
  2. *Segmentation*: 5-panel progressive visual workflow.
  3. *Confidence & XAI*: Viridis probability map, Inferno Shannon entropy heatmap, and uncertainty stratification.
  4. *Morphology*: Interactive histograms of area and circularity powered by Recharts.
  5. *Quantification*: Searchable and sortable nucleus morphometric data table.
- **`src/components/DownloadCenter.tsx`**: Export triggers for lossless `.npy` label arrays, `.csv` datasets, and publication PNGs.
- **`src/api.ts`**: HTTP client connecting with the FastAPI server.
- **`src/types.ts`**: TypeScript definitions matching API responses.

## Development

```bash
# Install dependencies
npm install

# Start Vite dev server on http://localhost:5173
npm run dev

# Production build
npm run build
```
