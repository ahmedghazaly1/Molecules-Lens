import { useEffect, useRef } from "react";

type Viewer = {
  addModel: (data: string, format: string) => void;
  setStyle: (sel: object, style: object) => void;
  zoomTo: () => void;
  render: () => void;
  clear: () => void;
};

declare global {
  interface Window {
    $3Dmol?: {
      createViewer: (element: HTMLElement, config: { backgroundColor: string }) => Viewer;
    };
  }
}

export function Molecule3D({ molblock }: { molblock: string }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = host.current;
    const api = window.$3Dmol;
    if (!node || !api) return;
    const viewer = api.createViewer(node, { backgroundColor: "white" });
    viewer.addModel(molblock, "mol");
    viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });
    viewer.zoomTo();
    viewer.render();
    return () => viewer.clear();
  }, [molblock]);

  return <div ref={host} className="mol3d" />;
}
