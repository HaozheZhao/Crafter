# Raster-to-SVG composition rules

## SVG generation rules

### Box/Shape Rules
- **Stroke width**: Academic figures use 2-3px borders. ALWAYS set stroke-width="2.5" minimum
- **Rounded corners**: Most boxes use rx="8" to rx="15". Match the original's corner radius
- **Fill colors**: Use K-means extracted colors, NOT LLM-approximated colors
- **Stroke colors**: Usually darker than fill. For green boxes: fill=#c4e4c2, stroke=#4a7a4a (not #1f291e)
- **Shadow**: Use feDropShadow with dx=1 dy=2 stdDeviation=2 flood-opacity=0.2 (subtle, not heavy)
- **Opacity**: All shapes should be opacity=1.0 unless explicitly transparent in original

### Text Rules
- **Font size**: Match bbox height. If bbox is 40px tall, font-size should be 32-36
- **Subscripts**: Render as SEPARATE <text> elements, NOT Unicode subscripts:
  - Base: <text x="100" y="200" font-size="36" font-style="italic">x</text>
  - Sub:  <text x="128" y="212" font-size="22" font-style="italic">1,t</text>
- **Font family**: Use "DejaVu Sans" (supports Unicode), NOT "Arial" or "Helvetica"
- **Text color**: Usually #000000 or #333333. Never faded/transparent
- **Bold titles**: Section headers ("Player 1's Local Planning Process") should be font-weight="bold"
- **Math italic**: Variable names (x, ψ, τ, d, φ) should be font-style="italic"

### Arrow Rules
- **Stroke width**: 1.5-2px (thin but visible)
- **Stroke color**: #808080 or #999999 (medium gray), NOT black
- **Arrowhead**: markerWidth="8" markerHeight="6" (small, proportional)
- **No overlap**: Arrows should connect BETWEEN boxes, never cross through box interiors
- **Dashed arrows**: stroke-dasharray="8 5" (visible dashes)
- **Curved arrows**: Use quadratic bezier Q, not straight lines through boxes

### Dashed Border Rules
- **Panel borders** (dashed-panel containers): stroke-width="2", stroke="#888888", stroke-dasharray="10 7"
- **NOT thin/faint**: Dashed borders must be clearly visible

### Background Rules
- **Colored backgrounds**: If the original has a colored background (not white), preserve it as a vector rect fill
- **Panel fills**: Light green for the first panel, light blue for the second — use extracted colors
- **No raster background**: Never embed the original image as a background layer

### Icon Embedding Rules
- **SVG format**: <image href="data:image/png;base64,{b64}" preserveAspectRatio="xMidYMid meet"/>
- **DrawIO format**: image=data:image/png,{b64} (NO ";base64" prefix)
- **Filter full-image detections**: Skip SAM3 detections covering >80% of image area
- **Morphological cleanup**: Close→Open→CC filter→Gaussian smooth on all masks

### DrawIO Conversion Rules
- **Convert from SVG**: Parse SVG elements → mxCells (ensures consistency)
- **Style mapping**: SVG rect → mxCell rounded=1;fillColor=X;strokeColor=Y;strokeWidth=Z
- **Image cells**: shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed
- **Edge cells**: endArrow=block;endFill=1 with sourcePoint/targetPoint
- **HTML in values**: Escape < > as &lt; &gt; for valid XML

## Iterative Refinement Process
1. Generate SVG from original + marked image
2. Render preview via cairosvg
3. Judge compares preview vs original (3 VLM ensemble)
4. If score < threshold: extract specific issues, feed back to LLM with current SVG
5. LLM fixes the SVG based on feedback
6. Repeat up to 3 iterations

## Detection Rules (SAM3)
- **Prompts per group**: background(panel,container), shape(rectangle,rounded rectangle), icon(icon,picture,logo), arrow(arrow,line,connector)
- **Thresholds**: background=0.25, shape=0.5, icon=0.5, arrow=0.45
- **Min area**: background=500, shape=200, icon=100, arrow=50
- **Filter**: Skip detections >80% of image area
- **Dedup**: Within-group IoU>0.7, cross-group priority-based, arrow IoU>0.85
