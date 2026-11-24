import process from "process/browser.js";

// Ketcher's browser dependencies use Node's process global during module loading.
globalThis.process = process;
globalThis.global = globalThis;

import("./main.tsx");
