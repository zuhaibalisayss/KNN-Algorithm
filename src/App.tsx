import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  knnClassify,
  knnRegress,
  generateExplanation,
  euclideanDistance,
  type DistanceMetric,
  type DataPoint,
  type PredictionResult,
} from './utils/knn';
import {
  mainDataset,
  defaultNewPoint,
  regressionDataset,
  defaultRegressionPoint,
  walkthroughDataset,
  walkthroughNewPoint,
  kComparisonDataset,
} from './data/datasets';

// ===== Progress Bar =====
function ProgressBar() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollPercent = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      setProgress(scrollPercent);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return <div className="progress-bar" style={{ width: `${progress}%` }} role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100} />;
}

// ===== SVG Plot Component =====
function KNNPlot({
  data,
  newPoint,
  result,
  width = 500,
  height = 500,
  padding = 40,
  showConnections = true,
  draggable = false,
  onDrag,
  xRange = [0, 10] as [number, number],
  yRange = [0, 10] as [number, number],
}: {
  data: DataPoint[];
  newPoint: { x: number; y: number };
  result: PredictionResult;
  width?: number;
  height?: number;
  padding?: number;
  showConnections?: boolean;
  draggable?: boolean;
  onDrag?: (x: number, y: number) => void;
  xRange?: [number, number];
  yRange?: [number, number];
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const scaleX = (val: number) => padding + ((val - xRange[0]) / (xRange[1] - xRange[0])) * (width - 2 * padding);
  const scaleY = (val: number) => height - padding - ((val - yRange[0]) / (yRange[1] - yRange[0])) * (height - 2 * padding);
  const inverseX = (px: number) => xRange[0] + ((px - padding) / (width - 2 * padding)) * (xRange[1] - xRange[0]);
  const inverseY = (py: number) => yRange[0] + ((height - padding - py) / (height - 2 * padding)) * (yRange[1] - yRange[0]);

  const handlePointerDown = (e: React.PointerEvent) => {
    if (!draggable) return;
    setIsDragging(true);
    (e.target as Element).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging || !svgRef.current || !onDrag) return;
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const py = ((e.clientY - rect.top) / rect.height) * height;
    const x = Math.max(xRange[0], Math.min(xRange[1], inverseX(px)));
    const y = Math.max(yRange[0], Math.min(yRange[1], inverseY(py)));
    onDrag(Math.round(x * 10) / 10, Math.round(y * 10) / 10);
  };

  const handlePointerUp = () => setIsDragging(false);

  const neighbourIds = new Set(result.neighbours.map(n => n.id));

  // Generate axes ticks
  const xTicks = [];
  const yTicks = [];
  for (let i = Math.ceil(xRange[0]); i <= Math.floor(xRange[1]); i++) {
    xTicks.push(i);
  }
  for (let i = Math.ceil(yRange[0]); i <= Math.floor(yRange[1]); i++) {
    yTicks.push(i);
  }

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto"
      style={{ maxHeight: '500px' }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      role="img"
      aria-label={`KNN plot showing ${data.length} training points and a new observation at (${newPoint.x.toFixed(1)}, ${newPoint.y.toFixed(1)})`}
    >
      {/* Grid */}
      {xTicks.map(t => (
        <line key={`gx-${t}`} x1={scaleX(t)} y1={padding} x2={scaleX(t)} y2={height - padding} stroke="#eee" strokeWidth={1} />
      ))}
      {yTicks.map(t => (
        <line key={`gy-${t}`} x1={padding} y1={scaleY(t)} x2={width - padding} y2={scaleY(t)} stroke="#eee" strokeWidth={1} />
      ))}

      {/* Axes */}
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#999" strokeWidth={1.5} />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#999" strokeWidth={1.5} />

      {/* Axis labels */}
      {xTicks.map(t => (
        <text key={`xt-${t}`} x={scaleX(t)} y={height - padding + 18} textAnchor="middle" fontSize={11} fill="#666">{t}</text>
      ))}
      {yTicks.map(t => (
        <text key={`yt-${t}`} x={padding - 10} y={scaleY(t) + 4} textAnchor="end" fontSize={11} fill="#666">{t}</text>
      ))}

      <text x={width / 2} y={height - 5} textAnchor="middle" fontSize={12} fill="#666">Feature X</text>
      <text x={12} y={height / 2} textAnchor="middle" fontSize={12} fill="#666" transform={`rotate(-90, 12, ${height / 2})`}>Feature Y</text>

      {/* Connections to neighbours */}
      {showConnections && result.neighbours.map(n => (
        <line
          key={`conn-${n.id}`}
          x1={scaleX(newPoint.x)}
          y1={scaleY(newPoint.y)}
          x2={scaleX(n.x)}
          y2={scaleY(n.y)}
          stroke={n.label === 'A' ? '#315A85' : '#C76A52'}
          strokeWidth={1.5}
          strokeDasharray="4,3"
          opacity={0.6}
        />
      ))}

      {/* Training points */}
      {data.map(point => {
        const isNeighbour = neighbourIds.has(point.id);
        const cx = scaleX(point.x);
        const cy = scaleY(point.y);

        if (point.label === 'A') {
          return (
            <circle
              key={point.id}
              cx={cx}
              cy={cy}
              r={isNeighbour ? 7 : 5}
              fill={isNeighbour ? '#315A85' : '#315A85'}
              fillOpacity={isNeighbour ? 1 : 0.6}
              stroke={isNeighbour ? '#1a3a5c' : 'none'}
              strokeWidth={isNeighbour ? 2 : 0}
            />
          );
        } else {
          return (
            <rect
              key={point.id}
              x={cx - (isNeighbour ? 6 : 4)}
              y={cy - (isNeighbour ? 6 : 4)}
              width={isNeighbour ? 12 : 8}
              height={isNeighbour ? 12 : 8}
              fill="#C76A52"
              fillOpacity={isNeighbour ? 1 : 0.6}
              stroke={isNeighbour ? '#8b4a39' : 'none'}
              strokeWidth={isNeighbour ? 2 : 0}
            />
          );
        }
      })}

      {/* New observation */}
      <g
        style={{ cursor: draggable ? 'grab' : 'default' }}
        onPointerDown={handlePointerDown}
      >
        <circle
          cx={scaleX(newPoint.x)}
          cy={scaleY(newPoint.y)}
          r={10}
          fill="white"
          stroke="#202124"
          strokeWidth={2.5}
        />
        <text
          x={scaleX(newPoint.x)}
          y={scaleY(newPoint.y) + 4}
          textAnchor="middle"
          fontSize={11}
          fontWeight="bold"
          fill="#202124"
        >?</text>
      </g>

      {/* Legend */}
      <g transform={`translate(${width - padding - 120}, ${padding + 5})`}>
        <circle cx={0} cy={0} r={5} fill="#315A85" />
        <text x={12} y={4} fontSize={11} fill="#333">Class A</text>
        <rect x={-4} y={12} width={8} height={8} fill="#C76A52" />
        <text x={12} y={20} fontSize={11} fill="#333">Class B</text>
        <circle cx={0} cy={36} r={6} fill="white" stroke="#202124" strokeWidth={2} />
        <text x={12} y={40} fontSize={11} fill="#333">New point</text>
      </g>
    </svg>
  );
}

// ===== Hero Section =====
function HeroSection() {
  return (
    <section className="section" style={{ minHeight: '90vh', display: 'flex', alignItems: 'center' }}>
      <div className="section-inner text-center" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <h1 style={{ marginBottom: '1rem' }}>K-Nearest Neighbours</h1>
        <p style={{ fontSize: 'clamp(1.1rem, 2vw, 1.4rem)', color: 'var(--color-text-secondary)', fontStyle: 'italic', maxWidth: '600px', marginBottom: '2rem' }}>
          How machines make predictions by learning from nearby examples.
        </p>
        <p style={{ maxWidth: '560px', textAlign: 'center' }}>
          A complete, interactive presentation that teaches you KNN from the ground up. 
          No advanced mathematics required — just curiosity and a willingness to learn.
        </p>
        <div style={{ display: 'flex', gap: '2rem', marginTop: '1.5rem', fontSize: '0.9rem', color: 'var(--color-text-secondary)' }}>
          <span>~20 min read</span>
          <span>•</span>
          <span>Interactive experiment included</span>
        </div>

        {/* Abstract data point diagram */}
        <svg viewBox="0 0 300 120" style={{ width: '280px', height: '110px', margin: '2.5rem 0' }} aria-hidden="true">
          <circle cx={60} cy={40} r={6} fill="#315A85" opacity={0.7} />
          <circle cx={80} cy={55} r={6} fill="#315A85" opacity={0.7} />
          <circle cx={45} cy={65} r={6} fill="#315A85" opacity={0.7} />
          <circle cx={70} cy={75} r={6} fill="#315A85" opacity={0.7} />
          <rect x={195} y={35} width={10} height={10} fill="#C76A52" opacity={0.7} />
          <rect x={215} y={55} width={10} height={10} fill="#C76A52" opacity={0.7} />
          <rect x={225} y={40} width={10} height={10} fill="#C76A52" opacity={0.7} />
          <rect x={200} y={70} width={10} height={10} fill="#C76A52" opacity={0.7} />
          <circle cx={150} cy={55} r={9} fill="white" stroke="#202124" strokeWidth={2} />
          <text x={150} y={59} textAnchor="middle" fontSize={10} fontWeight="bold" fill="#202124">?</text>
        </svg>

        {/* Scroll cue */}
        <div className="scroll-cue" style={{ marginTop: '2rem', color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
          <div style={{ width: '20px', height: '20px', margin: '0 auto', borderRight: '2px solid var(--color-text-secondary)', borderBottom: '2px solid var(--color-text-secondary)', transform: 'rotate(45deg)' }} />
        </div>
      </div>
    </section>
  );
}

// ===== Table of Contents =====
function TableOfContents() {
  const items = [
    { id: 'core-idea', label: 'The core idea' },
    { id: 'algorithm', label: 'How the algorithm works' },
    { id: 'distance', label: 'Understanding distance' },
    { id: 'experiment', label: 'The interactive experiment' },
    { id: 'choosing-k', label: 'Choosing K' },
    { id: 'classification-regression', label: 'Classification and regression' },
    { id: 'strengths-limitations', label: 'Strengths and limitations' },
    { id: 'applications', label: 'Applications' },
    { id: 'quiz', label: 'Test your understanding' },
    { id: 'summary', label: 'Final algorithm summary' },
  ];

  return (
    <nav className="section" style={{ padding: '2rem 1.5rem' }} aria-label="Table of contents">
      <div className="section-inner reading-width" style={{ margin: '0 auto' }}>
        <h3 style={{ marginTop: 0, fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-text-secondary)' }}>Contents</h3>
        <style>{`@media(min-width:600px){#toc-list{columns:2!important}}`}</style>
        <ol id="toc-list" style={{ listStyle: 'none', padding: 0, columns: '1', columnGap: '2rem' }}>
          {items.map((item, i) => (
            <li key={item.id} style={{ padding: '0.4rem 0', breakInside: 'avoid' }}>
              <a href={`#${item.id}`} className="toc-link">
                <span style={{ color: 'var(--color-text-secondary)', marginRight: '0.5rem', fontSize: '0.85em' }}>{i + 1}.</span>
                {item.label}
              </a>
            </li>
          ))}
        </ol>
      </div>
    </nav>
  );
}

// ===== Section 1: Core Idea =====
function CoreIdeaSection() {
  return (
    <section id="core-idea" className="section">
      <div className="section-inner">
        <h2>What is K-Nearest Neighbours?</h2>
        <div className="reading-width">
          <p>
            <strong>K-Nearest Neighbours</strong>, commonly called <strong>KNN</strong>, is a supervised machine learning algorithm. 
            It predicts the class or value of a new observation by examining the closest examples in its training data.
          </p>
          <p>
            In plain English: when KNN encounters something new, it looks at the most similar things it has already seen 
            and uses their known answers to make a prediction. It's like asking your nearest neighbours for advice.
          </p>

          <h3>An analogy</h3>
          <p>
            Imagine a new student whose exam result is unknown. We look at students with similar study habits, 
            attendance, and background. If most of the nearby students passed, we predict that the new student 
            will likely pass too.
          </p>
          <p style={{ color: 'var(--color-text-secondary)', fontStyle: 'italic', fontSize: '0.95em' }}>
            Important limitation: similarity does not guarantee identical outcomes. KNN makes a reasonable 
            prediction based on proximity, not a certainty.
          </p>

          <h3>The three words explained</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem', margin: '1.5rem 0' }}>
            <div style={{ padding: '1rem', borderLeft: '3px solid var(--color-accent)' }}>
              <strong>K</strong>
              <p style={{ margin: '0.5rem 0 0', fontSize: '0.95em' }}>The number of neighbours the algorithm considers when making a prediction.</p>
            </div>
            <div style={{ padding: '1rem', borderLeft: '3px solid var(--color-accent)' }}>
              <strong>Nearest</strong>
              <p style={{ margin: '0.5rem 0 0', fontSize: '0.95em' }}>The training observations that are closest to the new point, measured by a distance function.</p>
            </div>
            <div style={{ padding: '1rem', borderLeft: '3px solid var(--color-accent)' }}>
              <strong>Neighbours</strong>
              <p style={{ margin: '0.5rem 0 0', fontSize: '0.95em' }}>The existing labelled observations in the training dataset.</p>
            </div>
          </div>

          <h3>Key concepts</h3>
          <p>KNN is a <strong>supervised learning</strong> algorithm. This means it learns from training examples that already have known answers (labels).</p>
          
          <table className="data-table" style={{ maxWidth: '600px' }}>
            <thead>
              <tr><th>Term</th><th>Meaning</th></tr>
            </thead>
            <tbody>
              <tr><td><strong>Features</strong></td><td>The information used to describe each observation (e.g., study hours, attendance)</td></tr>
              <tr><td><strong>Label</strong></td><td>The answer we want to predict (e.g., pass or fail)</td></tr>
              <tr><td><strong>Training data</strong></td><td>Examples with known features and labels</td></tr>
              <tr><td><strong>New observation</strong></td><td>The example whose label is unknown</td></tr>
            </tbody>
          </table>

          {/* Simple diagram */}
          <div className="plot-container" style={{ maxWidth: '400px', margin: '2rem auto' }}>
            <svg viewBox="0 0 300 200" style={{ width: '100%' }}>
              <text x={150} y={20} textAnchor="middle" fontSize={12} fill="#666">Training data with known labels</text>
              <circle cx={60} cy={80} r={8} fill="#315A85" />
              <circle cx={80} cy={100} r={8} fill="#315A85" />
              <circle cx={50} cy={110} r={8} fill="#315A85" />
              <text x={63} y={140} textAnchor="middle" fontSize={11} fill="#315A85">Class A</text>
              <rect x={192} y={72} width={14} height={14} fill="#C76A52" />
              <rect x={212} y={92} width={14} height={14} fill="#C76A52" />
              <rect x={222} y={112} width={14} height={14} fill="#C76A52" />
              <text x={215} y={140} textAnchor="middle" fontSize={11} fill="#C76A52">Class B</text>
              <circle cx={150} cy={90} r={10} fill="white" stroke="#202124" strokeWidth={2.5} />
              <text x={150} y={94} textAnchor="middle" fontSize={12} fontWeight="bold" fill="#202124">?</text>
              <text x={150} y={170} textAnchor="middle" fontSize={11} fill="#666">New observation — which class?</text>
              <text x={150} y={185} textAnchor="middle" fontSize={10} fill="#999">KNN examines the nearest labelled points</text>
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}

// ===== Section 2: Algorithm Steps =====
function AlgorithmSection() {
  const [k, setK] = useState(3);
  const result = useMemo(() => knnClassify(walkthroughDataset, walkthroughNewPoint.x, walkthroughNewPoint.y, k, 'euclidean'), [k]);

  return (
    <section id="algorithm" className="section">
      <div className="section-inner">
        <h2>How does KNN make a prediction?</h2>
        <div className="reading-width">
          <p>The algorithm follows four clear steps. Let's walk through each one.</p>
        </div>

        <div style={{ display: 'grid', gap: '2rem', margin: '2rem 0' }}>
          {/* Step 1 */}
          <div style={{ padding: '1.5rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
            <h3 style={{ marginTop: 0, color: 'var(--color-accent)' }}>Step 1: Choose K</h3>
            <p>Decide how many neighbours will influence the prediction. Try changing K below:</p>
            <div className="slider-container">
              <label htmlFor="algo-k-slider">K =</label>
              <input id="algo-k-slider" type="range" min={1} max={7} value={k} onChange={e => setK(Number(e.target.value))} />
              <span style={{ fontWeight: 600, minWidth: '2ch' }}>{k}</span>
            </div>
          </div>

          {/* Step 2 */}
          <div style={{ padding: '1.5rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
            <h3 style={{ marginTop: 0, color: 'var(--color-accent)' }}>Step 2: Calculate distances</h3>
            <p>Measure the distance from the new observation to every training observation. We'll explore distance measures in detail in the next section.</p>
          </div>

          {/* Step 3 */}
          <div style={{ padding: '1.5rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
            <h3 style={{ marginTop: 0, color: 'var(--color-accent)' }}>Step 3: Select the K nearest</h3>
            <p>Sort all distances from smallest to largest, then select exactly K observations with the smallest distances. Points beyond the K-th nearest do not participate in this prediction.</p>
          </div>

          {/* Step 4 */}
          <div style={{ padding: '1.5rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
            <h3 style={{ marginTop: 0, color: 'var(--color-accent)' }}>Step 4: Predict</h3>
            <p>For <strong>classification</strong>: the majority class among the K neighbours wins.</p>
            <p>For <strong>regression</strong>: the prediction is the average of the neighbours' values.</p>
          </div>
        </div>

        {/* Live visualization */}
        <h3>Watch the algorithm work</h3>
        <p>Change K above and observe how the selected neighbourhood changes:</p>
        <div className="plot-container" style={{ maxWidth: '500px', margin: '1.5rem auto' }}>
          <KNNPlot
            data={walkthroughDataset}
            newPoint={walkthroughNewPoint}
            result={result}
            width={400}
            height={400}
          />
        </div>
        <div className="result-panel" style={{ maxWidth: '500px', margin: '1rem auto' }}>
          <p style={{ margin: 0, fontSize: '0.95em' }}>{generateExplanation(result)}</p>
        </div>
      </div>
    </section>
  );
}

// ===== Section 3: Distance =====
function DistanceSection() {
  return (
    <section id="distance" className="section">
      <div className="section-inner">
        <h2>How does KNN know which points are nearest?</h2>
        <div className="reading-width">
          <p>KNN needs a way to measure how similar two observations are. The most common approach is to calculate the geometric distance between them.</p>

          <h3>Euclidean distance</h3>
          <p>This is the straight-line distance between two points — the same concept you learned in geometry. In two dimensions:</p>
          
          <div className="formula-block">
            d = √((x₂ − x₁)² + (y₂ − y₁)²)
          </div>

          <p>Where:</p>
          <ul style={{ paddingLeft: '1.5rem' }}>
            <li><span className="inline-code">(x₁, y₁)</span> — coordinates of the first point</li>
            <li><span className="inline-code">(x₂, y₂)</span> — coordinates of the second point</li>
            <li><span className="inline-code">d</span> — the distance between them</li>
          </ul>

          <h3>Worked example</h3>
          <p>Calculate the distance between Point A at (1, 2) and Point B at (4, 6):</p>
          
          <div className="formula-block">
            <div>Step 1: Difference in x: 4 − 1 = 3</div>
            <div>Step 2: Difference in y: 6 − 2 = 4</div>
            <div>Step 3: Square the differences: 3² = 9, 4² = 16</div>
            <div>Step 4: Add them: 9 + 16 = 25</div>
            <div>Step 5: Square root: √25 = <strong>5</strong></div>
          </div>

          <p>The distance between (1, 2) and (4, 6) is <strong>5</strong>.</p>

          {/* Visual diagram */}
          <div className="plot-container" style={{ maxWidth: '350px', margin: '2rem auto' }}>
            <svg viewBox="0 0 300 280" style={{ width: '100%' }}>
              {/* Grid lines */}
              {[0,1,2,3,4,5].map(i => (
                <React.Fragment key={i}>
                  <line x1={50 + i * 40} y1={20} x2={50 + i * 40} y2={250} stroke="#eee" strokeWidth={1} />
                  <line x1={50} y1={250 - i * 30} x2={260} y2={250 - i * 30} stroke="#eee" strokeWidth={1} />
                </React.Fragment>
              ))}
              {/* Axes */}
              <line x1={50} y1={250} x2={260} y2={250} stroke="#999" strokeWidth={1.5} />
              <line x1={50} y1={20} x2={50} y2={250} stroke="#999" strokeWidth={1.5} />
              
              {/* Axis labels */}
              {[0,1,2,3,4,5].map(i => (
                <React.Fragment key={`xl-${i}`}>
                  <text x={50 + i * 40} y={268} textAnchor="middle" fontSize={10} fill="#999">{i}</text>
                </React.Fragment>
              ))}
              {[0,2,4,6].map(i => (
                <React.Fragment key={`yl-${i}`}>
                  <text x={42} y={250 - i * 30 + 4} textAnchor="end" fontSize={10} fill="#999">{i}</text>
                </React.Fragment>
              ))}
              
              {/* Points: A(1,2) and B(4,6) mapped to SVG coords */}
              {/* x: 50 + val*40, y: 250 - val*30 */}
              {/* A(1,2): x=90, y=190 */}
              {/* B(4,6): x=210, y=70 */}
              <circle cx={90} cy={190} r={7} fill="#315A85" />
              <text x={90} y={210} textAnchor="middle" fontSize={11} fill="#315A85" fontWeight="600">A (1, 2)</text>
              
              <circle cx={210} cy={70} r={7} fill="#C76A52" />
              <text x={210} y={58} textAnchor="middle" fontSize={11} fill="#C76A52" fontWeight="600">B (4, 6)</text>
              
              {/* Connection line (hypotenuse) */}
              <line x1={90} y1={190} x2={210} y2={70} stroke="#202124" strokeWidth={2} />
              
              {/* Horizontal difference (Δx = 3) */}
              <line x1={90} y1={190} x2={210} y2={190} stroke="#315A85" strokeWidth={1.5} strokeDasharray="4,3" />
              <text x={150} y={207} textAnchor="middle" fontSize={11} fill="#315A85">Δx = 3</text>
              
              {/* Vertical difference (Δy = 4) */}
              <line x1={210} y1={190} x2={210} y2={70} stroke="#C76A52" strokeWidth={1.5} strokeDasharray="4,3" />
              <text x={232} y={133} textAnchor="start" fontSize={11} fill="#C76A52">Δy = 4</text>
              
              {/* Right angle indicator */}
              <rect x={200} y={180} width={10} height={10} fill="none" stroke="#999" strokeWidth={1} />
              
              {/* Distance label */}
              <text x={138} y={125} textAnchor="middle" fontSize={13} fontWeight="bold" fill="#202124">d = 5</text>
            </svg>
          </div>

          <h3>Other distance measures</h3>
          <p>Briefly, two other common measures exist:</p>
          <ul style={{ paddingLeft: '1.5rem' }}>
            <li><strong>Manhattan distance</strong>: sums the absolute differences along each axis. |x₂ − x₁| + |y₂ − y₁|. Imagine walking along a grid of city blocks.</li>
            <li><strong>Minkowski distance</strong>: a generalization that includes both Euclidean and Manhattan as special cases.</li>
          </ul>
          <p>For this presentation, we focus on Euclidean distance, but the interactive experiment lets you compare both.</p>
        </div>
      </div>
    </section>
  );
}

// ===== Section 4: Interactive Experiment =====
function InteractiveExperiment() {
  const [k, setK] = useState(3);
  const [newPoint, setNewPoint] = useState({ ...defaultNewPoint });
  const [metric, setMetric] = useState<DistanceMetric>('euclidean');

  const result = useMemo(() => knnClassify(mainDataset, newPoint.x, newPoint.y, k, metric), [newPoint, k, metric]);

  const handleDrag = useCallback((x: number, y: number) => {
    setNewPoint({ x, y });
  }, []);

  const handleReset = () => {
    setK(3);
    setNewPoint({ ...defaultNewPoint });
    setMetric('euclidean');
  };

  return (
    <section id="experiment" className="section" style={{ background: 'rgba(49, 90, 133, 0.03)' }}>
      <div className="section-inner">
        <h2>Now, you control the algorithm.</h2>
        <div className="reading-width">
          <p>You've seen how KNN works. Now change its parameters and observe the prediction yourself. Every calculation is performed in real time — nothing is hardcoded.</p>
        </div>

        {/* Controls */}
        <div className="control-panel">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
            {/* K Slider */}
            <div>
              <label htmlFor="exp-k-slider" style={{ fontWeight: 600, display: 'block', marginBottom: '0.5rem' }}>
                K (number of neighbours)
              </label>
              <div className="slider-container">
                <input id="exp-k-slider" type="range" min={1} max={15} value={k} onChange={e => setK(Number(e.target.value))} aria-label="K value" />
                <span style={{ fontWeight: 700, fontSize: '1.2em', minWidth: '2.5ch', textAlign: 'center' }}>{k}</span>
              </div>
            </div>

            {/* Coordinate controls */}
            <div>
              <label style={{ fontWeight: 600, display: 'block', marginBottom: '0.5rem' }}>New observation position</label>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <div>
                  <label htmlFor="exp-x" style={{ fontSize: '0.85em', color: 'var(--color-text-secondary)' }}>X:</label>
                  <input
                    id="exp-x"
                    type="number"
                    min={0} max={10} step={0.1}
                    value={newPoint.x}
                    onChange={e => setNewPoint(p => ({ ...p, x: Math.max(0, Math.min(10, Number(e.target.value))) }))}
                    style={{ width: '70px', marginLeft: '0.5rem', padding: '0.3rem', border: '1px solid var(--color-border)', borderRadius: '3px' }}
                  />
                </div>
                <div>
                  <label htmlFor="exp-y" style={{ fontSize: '0.85em', color: 'var(--color-text-secondary)' }}>Y:</label>
                  <input
                    id="exp-y"
                    type="number"
                    min={0} max={10} step={0.1}
                    value={newPoint.y}
                    onChange={e => setNewPoint(p => ({ ...p, y: Math.max(0, Math.min(10, Number(e.target.value))) }))}
                    style={{ width: '70px', marginLeft: '0.5rem', padding: '0.3rem', border: '1px solid var(--color-border)', borderRadius: '3px' }}
                  />
                </div>
              </div>
              <p style={{ fontSize: '0.8em', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
                Or drag the ? point directly on the plot.
              </p>
            </div>

            {/* Distance metric */}
            <div>
              <label style={{ fontWeight: 600, display: 'block', marginBottom: '0.5rem' }}>Distance metric</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  className={`btn ${metric === 'euclidean' ? 'btn-primary' : ''}`}
                  onClick={() => setMetric('euclidean')}
                >Euclidean</button>
                <button
                  className={`btn ${metric === 'manhattan' ? 'btn-primary' : ''}`}
                  onClick={() => setMetric('manhattan')}
                >Manhattan</button>
              </div>
            </div>
          </div>

          <div style={{ marginTop: '1rem', textAlign: 'right' }}>
            <button className="btn" onClick={handleReset}>Reset all</button>
          </div>
        </div>

        {/* Main Plot */}
        <div className="plot-container" style={{ maxWidth: '600px', margin: '1.5rem auto' }}>
          <KNNPlot
            data={mainDataset}
            newPoint={newPoint}
            result={result}
            width={500}
            height={500}
            draggable
            onDrag={handleDrag}
          />
        </div>

        {/* Result Panel */}
        <div aria-live="polite" aria-atomic="true" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', margin: '1.5rem 0' }}>
          <div className="result-panel">
            <div className="result-item">
              <span className="result-label">Current K</span>
              <span className="result-value">{result.k}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Distance metric</span>
              <span className="result-value" style={{ textTransform: 'capitalize' }}>{result.metric}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Neighbours selected</span>
              <span className="result-value">{result.neighbours.length}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Class A votes</span>
              <span className="result-value class-a">{result.votes.A}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Class B votes</span>
              <span className="result-value class-b">{result.votes.B}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Predicted class</span>
              <span className="result-value" style={{ color: result.predictedClass === 'A' ? 'var(--color-accent)' : result.predictedClass === 'B' ? 'var(--color-class-b)' : 'inherit' }}>
                {result.predictedClass === 'tie' ? 'Tie' : `Class ${result.predictedClass}`}
              </span>
            </div>
            <div className="result-item">
              <span className="result-label">Vote share</span>
              <span className="result-value">{result.confidence}</span>
            </div>
          </div>

          <div>
            <h3 style={{ marginTop: 0 }}>Live explanation</h3>
            <p style={{ background: 'var(--color-code-bg)', padding: '1rem', borderRadius: '4px', border: '1px solid var(--color-border)' }}>
              {generateExplanation(result)}
            </p>
            <p style={{ fontSize: '0.85em', color: 'var(--color-text-secondary)', marginTop: '0.5rem' }}>
              Note: vote share is not a calibrated probability. It simply reflects the proportion of neighbours belonging to the winning class.
            </p>
          </div>
        </div>

        {/* Neighbour Table */}
        <h3>Selected neighbours (sorted by distance)</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Class</th>
                <th>Coordinates</th>
                <th>Distance</th>
                <th>In winning class?</th>
              </tr>
            </thead>
            <tbody>
              {result.neighbours.map(n => (
                <tr key={n.id}>
                  <td>{n.id}</td>
                  <td style={{ color: n.label === 'A' ? 'var(--color-accent)' : 'var(--color-class-b)', fontWeight: 600 }}>
                    {n.label === 'A' ? 'A ●' : 'B ■'}
                  </td>
                  <td>({n.x.toFixed(1)}, {n.y.toFixed(1)})</td>
                  <td>{n.distance.toFixed(3)}</td>
                  <td>{n.isWinningClass ? '✓ Yes' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Guided experiments */}
        <div style={{ marginTop: '2rem', padding: '1.5rem', border: '1px solid var(--color-border)', borderRadius: '6px', background: 'rgba(49, 90, 133, 0.02)' }}>
          <h3 style={{ marginTop: 0 }}>Guided experiments</h3>
          
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div>
              <strong>Experiment A: Change K</strong>
              <p style={{ margin: '0.25rem 0 0', fontSize: '0.95em' }}>
                Move K from 1 to 9. Notice how the selected neighbourhood grows and the prediction may change. 
                What happens at K = 1 vs K = 9?
              </p>
            </div>
            <div>
              <strong>Experiment B: Move the new point</strong>
              <p style={{ margin: '0.25rem 0 0', fontSize: '0.95em' }}>
                Drag the ? point toward each class cluster. Observe where the prediction switches from A to B. 
                Can you find the decision boundary?
              </p>
            </div>
            <div>
              <strong>Experiment C: Change the distance metric</strong>
              <p style={{ margin: '0.25rem 0 0', fontSize: '0.95em' }}>
                Switch between Euclidean and Manhattan for the same point position. 
                Do the selected neighbours change? Does the prediction change?
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ===== Section 5: Choosing K =====
function ChoosingKSection() {
  const kValues = [1, 5, 15];
  const testPoint = { x: 5, y: 5 };

  return (
    <section id="choosing-k" className="section">
      <div className="section-inner">
        <h2>What happens when K changes?</h2>
        <div className="reading-width">
          <p>The choice of K profoundly affects how KNN behaves. This is known as the <strong>bias-variance trade-off</strong>.</p>

          <h3>Small K (e.g., K = 1)</h3>
          <ul style={{ paddingLeft: '1.5rem' }}>
            <li>Uses very few neighbours</li>
            <li>Can capture fine-grained local patterns</li>
            <li>Highly sensitive to noise and outliers</li>
            <li>May produce unstable, jagged predictions</li>
          </ul>

          <h3>Large K (e.g., K = 15)</h3>
          <ul style={{ paddingLeft: '1.5rem' }}>
            <li>Uses many neighbours</li>
            <li>Produces smoother decision boundaries</li>
            <li>May overlook important local patterns</li>
            <li>Can become biased toward the majority class</li>
          </ul>

          <p>
            An extremely small K overfits to noise. An extremely large K oversimplifies. 
            Choosing K is a model-selection decision, often made through cross-validation.
          </p>
        </div>

        {/* K comparison */}
        <h3>Same data, different K values</h3>
        <p>Observe how the neighbourhood changes for the same test point:</p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', margin: '1.5rem 0' }}>
          {kValues.map(kv => {
            const res = knnClassify(kComparisonDataset, testPoint.x, testPoint.y, kv, 'euclidean');
            return (
              <div key={kv}>
                <div style={{ textAlign: 'center', fontWeight: 600, marginBottom: '0.5rem' }}>K = {kv}</div>
                <div className="plot-container">
                  <KNNPlot
                    data={kComparisonDataset}
                    newPoint={testPoint}
                    result={res}
                    width={250}
                    height={250}
                    padding={25}
                    showConnections={true}
                  />
                </div>
                <div style={{ textAlign: 'center', marginTop: '0.5rem', fontSize: '0.9em' }}>
                  <span className="class-a">A: {res.votes.A}</span> | <span className="class-b">B: {res.votes.B}</span>
                  <br />
                  <strong>Predicted: {res.predictedClass === 'tie' ? 'Tie' : `Class ${res.predictedClass}`}</strong>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ===== Section 6: Classification vs Regression =====
function ClassRegSection() {
  const [mode, setMode] = useState<'classification' | 'regression'>('classification');
  const [k, setK] = useState(3);

  const classResult = useMemo(() => knnClassify(mainDataset, defaultNewPoint.x, defaultNewPoint.y, k, 'euclidean'), [k]);
  const regResult = useMemo(() => knnRegress(regressionDataset, defaultRegressionPoint.x, defaultRegressionPoint.y, k, 'euclidean'), [k]);

  return (
    <section id="classification-regression" className="section">
      <div className="section-inner">
        <h2>KNN can solve two kinds of problems.</h2>
        <div className="reading-width">
          <p>The KNN principle is the same in both cases — examine nearby examples. What changes is how you turn those neighbours into a prediction.</p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', margin: '1.5rem 0' }}>
            <div style={{ padding: '1.25rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
              <h3 style={{ marginTop: 0, color: 'var(--color-accent)' }}>Classification</h3>
              <p>Predicts a <strong>category</strong>.</p>
              <p style={{ fontSize: '0.9em', color: 'var(--color-text-secondary)' }}>Example: Is this email spam or not spam?</p>
              <p>Uses <strong>majority voting</strong> among the K nearest labelled examples.</p>
            </div>
            <div style={{ padding: '1.25rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
              <h3 style={{ marginTop: 0, color: 'var(--color-class-b)' }}>Regression</h3>
              <p>Predicts a <strong>numerical value</strong>.</p>
              <p style={{ fontSize: '0.9em', color: 'var(--color-text-secondary)' }}>Example: What is this house's price?</p>
              <p>Calculates the <strong>mean</strong> of the K nearest neighbours' target values.</p>
            </div>
          </div>
        </div>

        {/* Toggle */}
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', justifyContent: 'center' }}>
          <button className={`btn ${mode === 'classification' ? 'btn-primary' : ''}`} onClick={() => setMode('classification')}>Classification</button>
          <button className={`btn ${mode === 'regression' ? 'btn-primary' : ''}`} onClick={() => setMode('regression')}>Regression</button>
        </div>

        <div className="slider-container" style={{ maxWidth: '400px', margin: '0 auto 1.5rem' }}>
          <label htmlFor="cr-k-slider">K =</label>
          <input id="cr-k-slider" type="range" min={1} max={7} value={k} onChange={e => setK(Number(e.target.value))} />
          <span style={{ fontWeight: 600 }}>{k}</span>
        </div>

        {mode === 'classification' ? (
          <div style={{ maxWidth: '500px', margin: '0 auto' }}>
            <div className="plot-container">
              <KNNPlot data={mainDataset} newPoint={defaultNewPoint} result={classResult} width={400} height={400} />
            </div>
            <div className="result-panel" style={{ marginTop: '1rem' }}>
              <div className="result-item">
                <span className="result-label">Predicted class</span>
                <span className="result-value">{classResult.predictedClass === 'tie' ? 'Tie' : `Class ${classResult.predictedClass}`}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Votes</span>
                <span className="result-value">A: {classResult.votes.A}, B: {classResult.votes.B}</span>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ maxWidth: '500px', margin: '0 auto' }}>
            <div className="plot-container">
              <svg viewBox="0 0 400 400" style={{ width: '100%' }}>
                {/* Regression points */}
                {regressionDataset.map((p, i) => {
                  const cx = 40 + (p.x / 10) * 320;
                  const cy = 360 - (p.y / 10) * 320;
                  const size = 4 + (p.target / 400) * 10;
                  return (
                    <g key={i}>
                      <circle cx={cx} cy={cy} r={size} fill="var(--color-accent)" opacity={0.6} />
                      <text x={cx} y={cy - size - 3} textAnchor="middle" fontSize={9} fill="#666">{p.target}</text>
                    </g>
                  );
                })}
                {/* New point */}
                <circle cx={40 + (defaultRegressionPoint.x / 10) * 320} cy={360 - (defaultRegressionPoint.y / 10) * 320} r={10} fill="white" stroke="#202124" strokeWidth={2.5} />
                <text x={40 + (defaultRegressionPoint.x / 10) * 320} y={360 - (defaultRegressionPoint.y / 10) * 320 + 4} textAnchor="middle" fontSize={11} fontWeight="bold">?</text>
                {/* Axes */}
                <line x1={40} y1={360} x2={360} y2={360} stroke="#999" strokeWidth={1.5} />
                <line x1={40} y1={40} x2={40} y2={360} stroke="#999" strokeWidth={1.5} />
                <text x={200} y={390} textAnchor="middle" fontSize={11} fill="#666">Feature X</text>
                <text x={15} y={200} textAnchor="middle" fontSize={11} fill="#666" transform="rotate(-90, 15, 200)">Feature Y</text>
              </svg>
            </div>
            <div className="result-panel" style={{ marginTop: '1rem' }}>
              <div className="result-item">
                <span className="result-label">Predicted value</span>
                <span className="result-value">{regResult.predictedValue.toFixed(1)}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Neighbour values</span>
                <span className="result-value">{regResult.neighbours.map(n => n.target).join(', ')}</span>
              </div>
              <div className="result-item">
                <span className="result-label">Mean calculation</span>
                <span className="result-value" style={{ fontSize: '0.9em' }}>
                  ({regResult.neighbours.map(n => n.target).join(' + ')}) / {k} = {regResult.predictedValue.toFixed(1)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// ===== Section 7: Strengths and Limitations =====
function StrengthsSection() {
  return (
    <section id="strengths-limitations" className="section">
      <div className="section-inner">
        <h2>Where does KNN work well, and where does it struggle?</h2>
        <div className="reading-width">
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ color: 'var(--color-accent)' }}>Advantages</th>
                  <th style={{ color: 'var(--color-class-b)' }}>Limitations</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Simple to understand and explain</td>
                  <td>Prediction can be slow on large datasets</td>
                </tr>
                <tr>
                  <td>Easy to implement</td>
                  <td>Must store all training observations</td>
                </tr>
                <tr>
                  <td>Works for both classification and regression</td>
                  <td>Sensitive to irrelevant features</td>
                </tr>
                <tr>
                  <td>No explicit training phase (lazy learner)</td>
                  <td>Sensitive to feature scaling</td>
                </tr>
                <tr>
                  <td>Can capture nonlinear local patterns</td>
                  <td>Struggles in high-dimensional spaces</td>
                </tr>
                <tr>
                  <td>Naturally adapts to multi-class problems</td>
                  <td>Performance depends heavily on K and the distance metric</td>
                </tr>
                <tr>
                  <td></td>
                  <td>Can be affected by class imbalance</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3>Why feature scaling matters</h3>
          <p>Consider two features:</p>
          <ul style={{ paddingLeft: '1.5rem' }}>
            <li><strong>Age</strong>: ranges from about 1 to 100</li>
            <li><strong>Income</strong>: ranges from about 1,000 to 1,000,000</li>
          </ul>
          <p>
            With Euclidean distance, the income feature will dominate the calculation because its values are much larger numerically. 
            A difference of 10,000 in income will overwhelm a difference of 10 years in age, even if age is more important for the prediction.
          </p>
          <p>
            <strong>Standardization</strong> (scaling features to have zero mean and unit variance) is a common solution. 
            However, not every dataset requires the same preprocessing — the appropriate approach depends on the data.
          </p>

          <h3>The curse of dimensionality</h3>
          <p>
            As the number of features grows, distances between points become less meaningful. 
            In very high-dimensional spaces, all points tend to be roughly equidistant from each other, 
            making it difficult for KNN to distinguish "near" from "far." This is why feature selection 
            and dimensionality reduction can be important when using KNN.
          </p>
        </div>
      </div>
    </section>
  );
}

// ===== Section 8: Applications =====
function ApplicationsSection() {
  const apps = [
    {
      title: 'Recommendation and similarity search',
      desc: 'Find items similar to a user\'s preferences by measuring distance in feature space. KNN can suggest products, movies, or content based on what similar users liked.'
    },
    {
      title: 'Pattern recognition',
      desc: 'Classify handwritten digits, facial expressions, or other patterns by comparing new samples to labelled examples in feature space.'
    },
    {
      title: 'Document classification',
      desc: 'Categorize documents by topic based on their word frequencies or embeddings. A new document is classified by the topics of its most similar documents.'
    },
    {
      title: 'Basic anomaly detection',
      desc: 'Flag observations that are far from their nearest neighbours as potentially unusual. If a point\'s K nearest neighbours are all very distant, it may be an anomaly.'
    },
    {
      title: 'Predicting numerical values',
      desc: 'Estimate continuous values like prices, temperatures, or scores by averaging the values of the most similar known observations.'
    }
  ];

  return (
    <section id="applications" className="section">
      <div className="section-inner">
        <h2>Where can KNN be used?</h2>
        <div className="reading-width">
          <p>KNN is a versatile algorithm. Here are five areas where it is commonly applied. Note that KNN is one possible approach in each area — not necessarily the best for every production system.</p>
        </div>
        <div style={{ display: 'grid', gap: '1.5rem', margin: '2rem 0' }}>
          {apps.map((app, i) => (
            <div key={i} style={{ padding: '1.25rem', borderLeft: '3px solid var(--color-accent)', background: 'rgba(49, 90, 133, 0.02)' }}>
              <h3 style={{ marginTop: 0, marginBottom: '0.5rem' }}>{i + 1}. {app.title}</h3>
              <p style={{ margin: 0, fontSize: '0.95em' }}>{app.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ===== Section 9: Quiz =====
function QuizSection() {
  const questions = [
    {
      q: 'What does K represent in KNN?',
      options: [
        'The number of features in the dataset',
        'The number of neighbours considered for prediction',
        'The number of training iterations',
        'The number of classes in the problem'
      ],
      correct: 1,
      explanation: 'K is the number of nearest neighbours the algorithm examines to make a prediction.'
    },
    {
      q: 'What determines which observations are "nearest"?',
      options: [
        'The alphabetical order of their labels',
        'The time they were added to the dataset',
        'A distance metric (e.g., Euclidean distance)',
        'The size of the dataset'
      ],
      correct: 2,
      explanation: 'A distance metric like Euclidean distance measures how close each training observation is to the new point.'
    },
    {
      q: 'How does standard KNN classification predict a class?',
      options: [
        'By averaging the features of all training points',
        'By majority voting among the K nearest neighbours',
        'By finding the single closest point only',
        'By fitting a linear model to the data'
      ],
      correct: 1,
      explanation: 'Standard KNN classification uses majority voting: the class with the most representatives among the K neighbours wins.'
    },
    {
      q: 'What typically happens when K is very small (e.g., K = 1)?',
      options: [
        'The model becomes very smooth and generalizes well',
        'The prediction is sensitive to noise and outliers',
        'All training points are used equally',
        'The algorithm runs faster'
      ],
      correct: 1,
      explanation: 'With K = 1, the prediction depends on a single neighbour, making it highly sensitive to noise and unusual observations.'
    },
    {
      q: 'Why can feature scaling matter for KNN?',
      options: [
        'It changes the number of classes',
        'Features with larger numerical ranges can dominate the distance calculation',
        'It determines the value of K',
        'It is required for the algorithm to run'
      ],
      correct: 1,
      explanation: 'Without scaling, features with large numerical ranges (e.g., income in thousands) can dominate Euclidean distance over features with small ranges (e.g., age).'
    },
    {
      q: 'How does KNN regression differ from KNN classification?',
      options: [
        'Regression uses a different distance metric',
        'Regression predicts a numerical value by averaging neighbours\' target values',
        'Regression does not use neighbours',
        'Regression only works with K = 1'
      ],
      correct: 1,
      explanation: 'KNN regression predicts a continuous value by computing the mean (or weighted mean) of the K nearest neighbours\' target values, instead of voting for a class.'
    }
  ];

  const [answers, setAnswers] = useState<(number | null)[]>(new Array(questions.length).fill(null));
  const [submitted, setSubmitted] = useState(false);
  const [showFeedback, setShowFeedback] = useState<boolean[]>(new Array(questions.length).fill(false));

  const handleSelect = (qIndex: number, optIndex: number) => {
    if (showFeedback[qIndex]) return;
    const newAnswers = [...answers];
    newAnswers[qIndex] = optIndex;
    setAnswers(newAnswers);
  };

  const handleSubmitQuestion = (qIndex: number) => {
    const newFeedback = [...showFeedback];
    newFeedback[qIndex] = true;
    setShowFeedback(newFeedback);
  };

  const handleSubmitAll = () => {
    setSubmitted(true);
    setShowFeedback(new Array(questions.length).fill(true));
  };

  const handleRetake = () => {
    setAnswers(new Array(questions.length).fill(null));
    setSubmitted(false);
    setShowFeedback(new Array(questions.length).fill(false));
  };

  const score = answers.reduce<number>((acc, a, i) => acc + (a === questions[i].correct ? 1 : 0), 0);

  return (
    <section id="quiz" className="section">
      <div className="section-inner">
        <h2>Can you explain KNN now?</h2>
        <div className="reading-width">
          <p>Test your understanding with these six questions. Select an answer and submit to see feedback.</p>
        </div>

        <div style={{ display: 'grid', gap: '2rem', margin: '2rem 0' }}>
          {questions.map((q, qi) => (
            <div key={qi} style={{ padding: '1.5rem', border: '1px solid var(--color-border)', borderRadius: '6px' }}>
              <p style={{ fontWeight: 600, marginBottom: '1rem' }}>
                <span style={{ color: 'var(--color-text-secondary)', marginRight: '0.5rem' }}>Q{qi + 1}.</span>
                {q.q}
              </p>
              {q.options.map((opt, oi) => {
                let className = 'quiz-option';
                if (showFeedback[qi]) {
                  if (oi === q.correct) className += ' correct';
                  else if (oi === answers[qi] && oi !== q.correct) className += ' incorrect';
                } else if (answers[qi] === oi) {
                  className += ' selected';
                }
                return (
                  <button
                    key={oi}
                    className={className}
                    onClick={() => handleSelect(qi, oi)}
                    disabled={showFeedback[qi]}
                    aria-pressed={answers[qi] === oi}
                  >
                    {opt}
                  </button>
                );
              })}
              {!showFeedback[qi] && answers[qi] !== null && (
                <button className="btn" style={{ marginTop: '0.75rem' }} onClick={() => handleSubmitQuestion(qi)}>
                  Check answer
                </button>
              )}
              {showFeedback[qi] && (
                <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'var(--color-code-bg)', borderRadius: '4px', fontSize: '0.9em' }}>
                  <strong>{answers[qi] === q.correct ? '✓ Correct!' : '✗ Not quite.'}</strong> {q.explanation}
                </div>
              )}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {!submitted ? (
            <button className="btn btn-primary" onClick={handleSubmitAll}>Submit all answers</button>
          ) : (
            <>
              <div style={{ fontSize: '1.2em', fontWeight: 600 }}>
                Score: {score} / {questions.length}
              </div>
              <button className="btn" onClick={handleRetake}>Retake quiz</button>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

// ===== Section 10: Summary =====
function SummarySection() {
  const [copied, setCopied] = useState(false);
  const linkUrl = typeof window !== 'undefined' ? window.location.href.split('#')[0] + '#experiment' : '#experiment';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(linkUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: select from a temporary input
      const input = document.createElement('input');
      input.value = linkUrl;
      document.body.appendChild(input);
      input.select();
      document.execCommand('copy');
      document.body.removeChild(input);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Complete numerical example
  const exampleData: DataPoint[] = [
    { id: 'E1', x: 1, y: 1, label: 'A' },
    { id: 'E2', x: 2, y: 2, label: 'A' },
    { id: 'E3', x: 1.5, y: 3, label: 'A' },
    { id: 'E4', x: 5, y: 5, label: 'B' },
    { id: 'E5', x: 6, y: 6, label: 'B' },
    { id: 'E6', x: 5.5, y: 4, label: 'B' },
  ];
  const exampleNew = { x: 3, y: 3 };
  const exampleResult = knnClassify(exampleData, exampleNew.x, exampleNew.y, 3, 'euclidean');

  return (
    <section id="summary" className="section">
      <div className="section-inner">
        <h2>You now understand KNN.</h2>
        <div className="reading-width">
          <p>Let's consolidate everything you've learned into a complete algorithm summary.</p>

          <h3>The algorithm in seven steps</h3>
          <ol style={{ paddingLeft: '1.5rem' }}>
            <li style={{ marginBottom: '0.5rem' }}><strong>Collect labelled training data.</strong> Each example has features and a known label.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Choose K.</strong> Decide how many neighbours will influence the prediction.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Select a distance metric.</strong> Euclidean distance is the most common choice.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Calculate distances</strong> from the new observation to every training observation.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Sort and select.</strong> Order by distance and pick the K nearest neighbours.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Predict.</strong> Use majority voting (classification) or mean (regression).</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>Evaluate.</strong> Adjust K, features, or preprocessing as needed.</li>
          </ol>

          <h3>A complete numerical example</h3>
          <p>Given training data and a new observation at (3, 3), with K = 3 and Euclidean distance:</p>

          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ maxWidth: '500px' }}>
              <thead>
                <tr><th>Point</th><th>Coordinates</th><th>Class</th><th>Distance to (3,3)</th></tr>
              </thead>
              <tbody>
                {exampleData.map(p => {
                  const d = euclideanDistance(exampleNew.x, exampleNew.y, p.x, p.y);
                  const isNeighbour = exampleResult.neighbours.some(n => n.id === p.id);
                  return (
                    <tr key={p.id} style={{ background: isNeighbour ? 'rgba(49, 90, 133, 0.05)' : undefined }}>
                      <td>{p.id}</td>
                      <td>({p.x}, {p.y})</td>
                      <td style={{ color: p.label === 'A' ? 'var(--color-accent)' : 'var(--color-class-b)', fontWeight: 600 }}>{p.label}</td>
                      <td>{d.toFixed(3)} {isNeighbour && '← selected'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="formula-block">
            <div><strong>Distances calculated:</strong></div>
            {exampleData.map(p => {
              const d = euclideanDistance(exampleNew.x, exampleNew.y, p.x, p.y);
              return <div key={p.id}>{p.id}: √((3−{p.x})² + (3−{p.y})²) = {d.toFixed(3)}</div>;
            })}
            <div style={{ marginTop: '0.5rem' }}><strong>Sorted by distance, K = 3 selected:</strong></div>
            {exampleResult.neighbours.map(n => (
              <div key={n.id}>{n.id} (Class {n.label}): {n.distance.toFixed(3)}</div>
            ))}
            <div style={{ marginTop: '0.5rem' }}>
              <strong>Vote:</strong> A = {exampleResult.votes.A}, B = {exampleResult.votes.B}
            </div>
            <div>
              <strong>Predicted class: {exampleResult.predictedClass === 'tie' ? 'Tie' : `Class ${exampleResult.predictedClass}`}</strong>
            </div>
          </div>

          <h3>The final takeaway</h3>
          <div style={{ padding: '1.5rem', background: 'var(--color-code-bg)', borderRadius: '6px', border: '1px solid var(--color-border)', margin: '1.5rem 0' }}>
            <p style={{ fontWeight: 600, fontSize: '1.1em', margin: 0 }}>
              KNN predicts a new observation by examining the nearest examples in its training data.
            </p>
          </div>

          <p>The three most important decisions when using KNN:</p>
          <ol style={{ paddingLeft: '1.5rem' }}>
            <li>How many neighbours to use (K)</li>
            <li>How to measure distance (metric)</li>
            <li>How to turn neighbours into a prediction (voting or averaging)</li>
          </ol>

          <h3>Learning checklist</h3>
          <ul className="checklist">
            <li>I can define KNN.</li>
            <li>I can explain its four main steps.</li>
            <li>I understand Euclidean distance.</li>
            <li>I can explain the effect of K.</li>
            <li>I can distinguish classification from regression.</li>
            <li>I understand why feature scaling matters.</li>
            <li>I can interpret the interactive KNN experiment.</li>
          </ul>
        </div>

        {/* Shareable link */}
        <div style={{ marginTop: '3rem', padding: '2rem', border: '1px solid var(--color-border)', borderRadius: '6px', textAlign: 'center', background: 'rgba(49, 90, 133, 0.02)' }}>
          <h3 style={{ marginTop: 0 }}>Try the experiment yourself</h3>
          <p>Return to the interactive experiment. Change K, move the new point, and compare distance metrics.</p>
          <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap', alignItems: 'center' }}>
            <a href="#experiment" className="btn btn-primary">Go to experiment</a>
            <button className="btn" onClick={handleCopy}>
              {copied ? '✓ Copied!' : 'Copy link'}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

// ===== Main App =====
export default function App() {
  const [presentationMode, setPresentationMode] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && presentationMode) {
        setPresentationMode(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [presentationMode]);

  return (
    <div className={presentationMode ? 'presentation-mode' : ''}>
      <ProgressBar />

      {/* Presentation mode toggle */}
      <div style={{ position: 'fixed', bottom: '1.5rem', right: '1.5rem', zIndex: 100 }}>
        <button
          className="btn"
          onClick={() => setPresentationMode(!presentationMode)}
          style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
          aria-label={presentationMode ? 'Exit presentation mode' : 'Enter presentation mode'}
        >
          {presentationMode ? 'Normal mode' : 'Present'}
        </button>
      </div>

      <main>
        <HeroSection />
        <TableOfContents />
        <CoreIdeaSection />
        <AlgorithmSection />
        <DistanceSection />
        <InteractiveExperiment />
        <ChoosingKSection />
        <ClassRegSection />
        <StrengthsSection />
        <ApplicationsSection />
        <QuizSection />
        <SummarySection />
      </main>

      <footer style={{ padding: '2rem 1.5rem', textAlign: 'center', color: 'var(--color-text-secondary)', fontSize: '0.85em', borderTop: '1px solid var(--color-border)' }}>
        <p>K-Nearest Neighbours — An Interactive Learning Presentation</p>
        <p style={{ margin: '0.25rem 0 0' }}>Built for education. No tracking, no accounts, no dependencies on external services.</p>
      </footer>
    </div>
  );
}
