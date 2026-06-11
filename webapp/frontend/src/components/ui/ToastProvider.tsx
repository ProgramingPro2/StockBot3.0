import { create } from "zustand";
import { createPortal } from "react-dom";
import Toast, { type ToastData } from "./Toast";

type ToastStore = {
  toasts: ToastData[];
  add: (msg: string, type?: ToastData["type"]) => void;
  dismiss: (id: string) => void;
};

export const useToast = create<ToastStore>((set) => ({
  toasts: [],
  add: (message, type = "info") =>
    set((s) => ({
      toasts: [...s.toasts, { id: crypto.randomUUID(), message, type }],
    })),
  dismiss: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

export default function ToastProvider() {
  const { toasts, dismiss } = useToast();
  return createPortal(
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 w-80 pointer-events-none">
      {toasts.map((t) => (
        <Toast key={t.id} {...t} onDismiss={dismiss} />
      ))}
    </div>,
    document.body,
  );
}
