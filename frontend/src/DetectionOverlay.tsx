import type { Detection } from "./types";

export default function DetectionOverlay({
  objects,
}: {
  objects: Detection[];
}) {
  return (
    <div className="detection-overlay" aria-hidden="true">
      {objects.map((object, i) => {
        const [x1, y1, x2, y2] = object.box;
        const color = "#C8C4B8";
        return (
          <div
            key={`${object.label}-${i}`}
            className="bounding-box"
            style={{
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${(x2 - x1) * 100}%`,
              height: `${(y2 - y1) * 100}%`,
              borderColor: color,
            }}
          >
            <span
              style={{
                background: color,
                ...(y1 < 0.06 ? { transform: "none" } : {}),
              }}
            >
              {object.label} <b>{Math.round(object.confidence * 100)}%</b>
            </span>
          </div>
        );
      })}
    </div>
  );
}
