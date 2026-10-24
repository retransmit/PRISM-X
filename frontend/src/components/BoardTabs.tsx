import { BOARDS, BOARD_LABELS, type Board } from '../boards/board'
import './boards.css'

interface Props {
  active: Board
  onSelect: (board: Board) => void
}

export function BoardTabs({ active, onSelect }: Props) {
  return (
    <nav className="tabs" role="tablist" aria-label="Boards">
      {BOARDS.map((board) => (
        <button
          key={board}
          type="button"
          role="tab"
          id={`tab-${board}`}
          aria-selected={active === board}
          aria-controls={`board-${board}`}
          className="tabs__tab"
          onClick={() => onSelect(board)}
        >
          {BOARD_LABELS[board]}
        </button>
      ))}
    </nav>
  )
}
