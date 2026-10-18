# K-Nearest Neighbours — Interactive Learning Presentation

A complete, interactive educational website that teaches the K-Nearest Neighbours (KNN) algorithm from scratch. Built for classroom presentations and self-study.

## Overview

This is a single-page educational presentation that covers:

- What KNN is and how it works
- Euclidean and Manhattan distance calculations
- A fully interactive KNN experiment with real calculations
- The effect of choosing different K values
- Classification vs regression with KNN
- Advantages, limitations, and real-world applications
- An interactive knowledge quiz
- A complete algorithm summary

## Features

- **Interactive KNN Experiment**: Change K, move the test point, switch distance metrics — all with real-time calculations
- **Visual Learning**: SVG-based diagrams that update dynamically
- **Knowledge Quiz**: Six questions with immediate feedback
- **Presentation Mode**: Larger text for classroom projection
- **Fully Responsive**: Works on desktop, tablet, and mobile
- **No Dependencies**: No external APIs, no accounts, no tracking
- **Accessible**: Keyboard navigation, ARIA labels, reduced-motion support

## Local Setup

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The site will be available at `http://localhost:3000`.

### Build for Production

```bash
npm run build
```

The output will be in the `dist/` directory.

## Deploying to GitHub Pages

1. Create a new public GitHub repository
2. Push this project to the repository
3. Install dependencies and build:
   ```bash
   npm install
   npm run build
   ```
4. Push the `dist/` folder to a `gh-pages` branch:
   ```bash
   git subtree push --prefix dist origin gh-pages
   ```
   Or use the `gh-pages` npm package:
   ```bash
   npm install -D gh-pages
   npx gh-pages -d dist
   ```
5. In your repository settings, enable GitHub Pages with source set to the `gh-pages` branch
6. Your site will be available at `https://yourusername.github.io/repository-name/`

### Alternative: Using GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm install
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist
```

## Sharing the Deployed URL

Once deployed, share the GitHub Pages URL with classmates. The interactive experiment is accessible at the `#experiment` anchor:

```
https://yourusername.github.io/repository-name/#experiment
```

## Troubleshooting

### Build fails
- Ensure Node.js 18+ is installed: `node --version`
- Delete `node_modules` and `package-lock.json`, then run `npm install` again

### Styles not loading
- The project uses Tailwind CSS v4 with the Vite plugin. Ensure `@tailwindcss/vite` is in devDependencies.

### Interactive experiment not working
- Check the browser console for JavaScript errors
- Ensure JavaScript is enabled in the browser
- The experiment works entirely client-side — no server needed

### Presentation mode
- Click the "Present" button in the bottom-right corner
- Press Escape to exit presentation mode

## Project Structure

```
├── index.html          # HTML entry point
├── src/
│   ├── main.tsx        # React entry point
│   ├── App.tsx         # Main application with all sections
│   ├── index.css       # Tailwind + custom styles
│   ├── utils/
│   │   └── knn.ts      # KNN algorithm implementation
│   └── data/
│       └── datasets.ts # Deterministic datasets
├── package.json
├── vite.config.js
└── README.md
```

## Technology

- React 18
- Vite
- Tailwind CSS v4
- SVG for visualizations
- No external API dependencies
- No backend required

## License

Free to use for educational purposes.
