import Modal from "./Modal";

type Props = {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  title?: string;
  message: string;
  confirmLabel?: string;
  destructive?: boolean;
};

export default function ConfirmDialog({
  open, onConfirm, onCancel, title = "Confirm", message, confirmLabel = "Confirm", destructive = false,
}: Props) {
  return (
    <Modal open={open} onClose={onCancel} title={title} maxWidth="max-w-md">
      <p className="text-sm text-ink-muted">{message}</p>
      <div className="flex justify-end gap-2 mt-4">
        <button type="button" className="btn-secondary" onClick={onCancel}>Cancel</button>
        <button
          type="button"
          className={destructive ? "btn-danger" : "btn-primary"}
          onClick={onConfirm}
        >
          {confirmLabel}
        </button>
      </div>
    </Modal>
  );
}
