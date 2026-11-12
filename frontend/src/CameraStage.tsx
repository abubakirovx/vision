import { useEffect, useRef, useState, type ReactNode } from "react";

/** Video and boxes share exactly the same contained rectangle, including fullscreen. */
export default function CameraStage({
  width,
  height,
  children,
}: {
  width: number;
  height: number;
  children: ReactNode;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [bounds, setBounds] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      setBounds({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });
    if (host.current) observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  const ratio = width / height;
  const fittedWidth = Math.min(bounds.width, bounds.height * ratio);
  return (
    <div ref={host} className="camera-fit">
      <div
        className="camera-stage"
        style={{ width: fittedWidth, height: fittedWidth / ratio }}
      >
        {children}
      </div>
    </div>
  );
}
