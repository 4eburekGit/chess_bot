import numpy as np

def sq_name(x, y):
    return chr(ord('a') + x) + str(y + 1)

# базовый класс для всех фигур
class Figure:
    def __init__(self, board, x, y, color):
        self.board = board
        self.x = x
        self.y = y
        self.color = color
        self.alive = True
        self.has_moved = False
        self.initial_x = x
        self.promoted_from_pawn = False

    # геттеры
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def isAlive(self):
        return self.alive

    def getColor(self):
        return self.color

    # служебное
    def letter(self):
        return '?'

    # интерфейс
    def checkLegal(self, x, y):
        if not (0 <= x < 8 and 0 <= y < 8):
            return False
        if x == self.x and y == self.y:
            return False
        target = self.board.get(x, y)
        if target is not None and target.color == self.color:
            return False
        if not self._pattern(x, y):
            return False
        return not self.board.leaves_king_in_check(self, x, y)

    def _pattern(self, x, y):
        return False

    def move(self, x, y):
        if not self.checkLegal(x, y):
            return False
        self.board.apply_move(self, x, y)
        return True


# скользящие фигуры

class SlidingFigure(Figure):
    def directions(self):
        return []

    def _pattern(self, x, y):
        dx = x - self.x
        dy = y - self.y
        for ddx, ddy in self.directions():
            if ddx == 0:
                if dx != 0 or dy * ddy <= 0:
                    continue
                steps, ux, uy = abs(dy), 0, (1 if dy > 0 else -1)
            elif ddy == 0:
                if dy != 0 or dx * ddx <= 0:
                    continue
                steps, ux, uy = abs(dx), (1 if dx > 0 else -1), 0
            else:
                if dx * ddy != dy * ddx or dx * ddx <= 0:
                    continue
                steps = abs(dx)
                ux, uy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
            blocked = any(
                self.board.get(self.x + ux * i, self.y + uy * i) is not None
                for i in range(1, steps)
            )
            if not blocked:
                return True
        return False


class Rook(SlidingFigure):
    def letter(self):
        return 'R'

    def directions(self):
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def canCastle(self):
        return not self.has_moved


class Bishop(SlidingFigure):
    def letter(self):
        return 'B'

    def directions(self):
        return [(1, 1), (1, -1), (-1, 1), (-1, -1)]


class Queen(SlidingFigure):
    def letter(self):
        return 'Q'

    def directions(self):
        return [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]


# конь 

class Knight(Figure):
    def letter(self):
        return 'N'

    def _pattern(self, x, y):
        dx, dy = abs(x - self.x), abs(y - self.y)
        return (dx, dy) in [(1, 2), (2, 1)]


# король 

class King(Figure):
    def letter(self):
        return 'K'

    def _pattern(self, x, y):
        return max(abs(x - self.x), abs(y - self.y)) == 1

    def check(self):
        # проверка на шах
        opp = 'b' if self.color == 'w' else 'w'
        return self.board.is_square_attacked(self.x, self.y, opp)

    def canCastle(self, kingside):
        if self.has_moved or self.check():
            return False
        y = 0 if self.color == 'w' else 7
        if self.y != y or self.x != 4:
            return False
        rook_x = 7 if kingside else 0
        rook = self.board.get(rook_x, y)
        if not isinstance(rook, Rook) or not rook.alive \
                or rook.has_moved or rook.color != self.color:
            return False
        opp = 'b' if self.color == 'w' else 'w'
        if kingside:
            if any(self.board.get(c, y) is not None for c in (5, 6)):
                return False
            if self.board.is_square_attacked(5, y, opp):
                return False
            if self.board.is_square_attacked(6, y, opp):
                return False
        else:
            if any(self.board.get(c, y) is not None for c in (1, 2, 3)):
                return False
            if self.board.is_square_attacked(3, y, opp):
                return False
            if self.board.is_square_attacked(2, y, opp):
                return False
        return True

    def castle(self, kingside):
        if not self.canCastle(kingside):
            return False
        self.board.apply_castle(self, kingside)
        return True

    def move(self, x, y):
        if abs(x - self.x) == 2 and y == self.y:
            return self.castle(x > self.x)
        return super().move(x, y)


# пешка 

class Pawn(Figure):
    def __init__(self, board, x, y, color):
        super().__init__(board, x, y, color)
        self.initial_x = x
        self.promoted_to = None
        
    def letter(self):
        return ''

    def _pattern(self, x, y):
        d = 1 if self.color == 'w' else -1
        dx, dy = x - self.x, y - self.y
        target = self.board.get(x, y)
        # ход вперёд
        if dx == 0 and dy == d and target is None:
            return True
        # двойной ход с начальной позиции
        if dx == 0 and dy == 2 * d and not self.has_moved and target is None:
            if self.board.get(self.x, self.y + d) is None:
                return True
        # взятие по диагонали
        if abs(dx) == 1 and dy == d:
            if target is not None and target.color != self.color:
                return True
            # взятие на проходе
            if self.board.en_passant_target == (x, y):
                return True
        return False

    def promote(self, choice):
        # choice in [0,4): 0-Q 1-B 2-R 3-N.
        classes = [Queen, Bishop, Rook, Knight]
        letters = ['Q', 'B', 'R', 'N']
        self.promoted_to = letters[choice]
        new_piece = classes[choice](self.board, self.x, self.y, self.color)
        new_piece.has_moved = True
        new_piece.promoted_from_pawn = True

        self.board.grid[self.x, self.y] = new_piece
        self.board.figures.append(new_piece)
        self.alive = False
        return new_piece


# доска

class Board:
    def __init__(self, verbose):
        self.grid = np.empty((8, 8), dtype=object)
        self.grid[:] = None
        self.figures = []
        self.turn = 'w'
        self._active = True
        self.en_passant_target = None
        self.move_log = []
        self.verbose = verbose
        self.setup()

    # расстановка 
    def setup(self):
        for x in range(8):
            self._place(Pawn(self, x, 1, 'w'))
            self._place(Pawn(self, x, 6, 'b'))
        back = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for x, cls in enumerate(back):
            self._place(cls(self, x, 0, 'w'))
            self._place(cls(self, x, 7, 'b'))

    def _place(self, f):
        self.figures.append(f)
        self.grid[f.x, f.y] = f

    # базовые операции
    def get(self, x, y):
        if 0 <= x < 8 and 0 <= y < 8:
            return self.grid[x, y]
        return None

    def active(self):
        return self._active

    def getFigures(self):
        """Плоский массив 64 int: 8 пешек + 8 фигур на сторону,
        в порядке initial_x. Мёртвые — (-1,-1).
        Превращённая пешка показывает координаты своего слот-положения."""
        out = []
        for color in ('w', 'b'):
            side = [f for f in self.figures if f.color == color]
            pawns = sorted([f for f in side if isinstance(f, Pawn)],
                           key=lambda f: f.initial_x)
            others = sorted([f for f in side
                             if not isinstance(f, Pawn)
                             and not f.promoted_from_pawn],
                            key=lambda f: f.initial_x)
            for f in pawns[:8]:
                if f.alive:
                    out.extend([f.x, f.y])
                elif f.promoted_to is not None:
                    out.extend([f.x, f.y])
                else:
                    out.extend([-1, -1])
            for f in others[:8]:
                if f.alive:
                    out.extend([f.x, f.y])
                else:
                    out.extend([-1, -1])
        return np.array(out, dtype=int)

    def getPawnStates(self):
        state_map = {None: 0.1, 'Q': 0.3, 'B': 0.5, 'R': 0.7, 'N': 0.9}
        out = []
        for color in ('w', 'b'):
            pawns = sorted([f for f in self.figures
                            if f.color == color and isinstance(f, Pawn)],
                           key=lambda f: f.initial_x)
            for p in pawns[:8]:
                out.append(state_map.get(p.promoted_to, 0.1))
        return np.array(out, dtype=np.float32)

    def replace_piece(self, old, new):
        self.grid[new.x, new.y] = new
        idx = self.figures.index(old)
        self.figures[idx] = new

    def find_king(self, color):
        for f in self.figures:
            if isinstance(f, King) and f.color == color and f.alive:
                return f
        return None

    # атака и шах
    def is_square_attacked(self, x, y, by_color):
        for f in self.figures:
            if not f.alive or f.color != by_color:
                continue
            if isinstance(f, King):
                if max(abs(f.x - x), abs(f.y - y)) == 1:
                    return True
            elif isinstance(f, Pawn):
                d = 1 if f.color == 'w' else -1
                if abs(f.x - x) == 1 and (y - f.y) == d:
                    return True
            else:
                if f._pattern(x, y):
                    return True
        return False

    def leaves_king_in_check(self, figure, x, y):
        # симуляция хода для проверки на открытие шаха
        king = self.find_king(figure.color)
        if king is None:
            return False
        ox, oy = figure.x, figure.y
        target = self.grid[x, y]

        # сохраняем состояние для отката
        ep_cap = ep_pos = None
        if isinstance(figure, Pawn) and target is None \
                and self.en_passant_target == (x, y):
            cy = y - 1 if figure.color == 'w' else y + 1
            ep_cap = self.grid[x, cy]
            ep_pos = (x, cy)
            if ep_cap is not None:
                self.grid[x, cy] = None
                ep_cap.alive = False

        self.grid[ox, oy] = None
        self.grid[x, y] = figure
        figure.x, figure.y = x, y
        if target is not None:
            target.alive = False

        opp = 'b' if figure.color == 'w' else 'w'
        result = self.is_square_attacked(king.x, king.y, opp)

        # откат
        self.grid[ox, oy] = figure
        figure.x, figure.y = ox, oy
        self.grid[x, y] = target
        if target is not None:
            target.alive = True
        if ep_cap is not None:
            self.grid[ep_pos[0], ep_pos[1]] = ep_cap
        return result

    # нотация
    def _build_notation(self, figure, x, y):
        target = self.grid[x, y]
        is_capture = target is not None

        if isinstance(figure, Pawn):
            if not is_capture and self.en_passant_target == (x, y):
                is_capture = True
            prefix = chr(ord('a') + figure.x) + 'x' if is_capture else ''
            return prefix + sq_name(x, y)

        # шахматная фигура
        letter = figure.letter()
        # расшифровка
        same = [f for f in self.figures
                if f is not figure and f.alive and f.color == figure.color
                and type(f) is type(figure) and f.checkLegal(x, y)]
        dis = ''
        if same:
            if not any(f.x == figure.x for f in same):
                dis = chr(ord('a') + figure.x)
            elif not any(f.y == figure.y for f in same):
                dis = str(figure.y + 1)
            else:
                dis = sq_name(figure.x, figure.y)
        cap = 'x' if is_capture else ''
        return f"{letter}{dis}{cap}{sq_name(x, y)}"

    def _check_suffix(self):
        side = self.turn  # уже после смены хода
        king = self.find_king(side)
        if king is None:
            return ''
        opp = 'b' if side == 'w' else 'w'
        if not self.is_square_attacked(king.x, king.y, opp):
            return ''
        return '+' if self._has_any_legal_move(side) else '#'

    def _has_any_legal_move(self, color):
        for f in self.figures:
            if not f.alive or f.color != color:
                continue
            for x in range(8):
                for y in range(8):
                    if f.checkLegal(x, y):
                        return True
            if isinstance(f, King):
                for side in (True, False):
                    if f.canCastle(side):
                        return True
        return False

    # применение ходов
    def apply_move(self, figure, x, y):
        notation = self._build_notation(figure, x, y)
        ox, oy = figure.x, figure.y
        target = self.grid[x, y]

        # взятие на проходе (omg google en passant)
        if isinstance(figure, Pawn) and target is None \
                and self.en_passant_target == (x, y):
            cy = y - 1 if figure.color == 'w' else y + 1
            cap = self.grid[x, cy]
            if cap is not None:
                cap.alive = False
                self.grid[x, cy] = None

        if target is not None:
            target.alive = False

        self.grid[ox, oy] = None
        self.grid[x, y] = figure
        figure.x, figure.y = x, y
        figure.has_moved = True

        # новый en passant target
        self.en_passant_target = None
        if isinstance(figure, Pawn) and abs(y - oy) == 2:
            self.en_passant_target = (x, (y + oy) // 2)

        # ход переходит другой стороне
        self.turn = 'b' if self.turn == 'w' else 'w'

        # превращение пешки — ставим заглушку, подставим итог в Board
        is_promo = isinstance(figure, Pawn) and (
            (figure.color == 'w' and y == 7) or
            (figure.color == 'b' and y == 0)
        )
        if is_promo:
            notation += '=?'

        suffix = self._check_suffix()
        notation += suffix
        self.move_log.append(notation)
        if suffix == '#':
            self._active = False
        return notation

    def apply_castle(self, king, kingside):
        y = king.y
        notation = "O-O" if kingside else "O-O-O"
        if kingside:
            rook = self.grid[7, y]
            self.grid[king.x, y] = None
            self.grid[7, y] = None
            self.grid[6, y] = king
            self.grid[5, y] = rook
            king.x, rook.x = 6, 5
        else:
            rook = self.grid[0, y]
            self.grid[king.x, y] = None
            self.grid[0, y] = None
            self.grid[2, y] = king
            self.grid[3, y] = rook
            king.x, rook.x = 2, 3
        king.has_moved = True
        rook.has_moved = True
        self.turn = 'b' if self.turn == 'w' else 'w'
        suffix = self._check_suffix()
        notation += suffix
        self.move_log.append(notation)
        if suffix == '#':
            self._active = False
        return notation

    # вывод доски
    def print_board(self):
        if not self.verbose:            # для обучения
            return
        for y in range(7, -1, -1):
            row = f"{y + 1} {y} "
            for x in range(8):
                f = self.grid[x, y]
                if f is None:
                    row += ". "
                else:
                    letter = f.letter() if f.letter() else 'P'
                    row += (letter if f.color == 'w' else letter.lower()) + ' '
            print(row)
        print("    0 1 2 3 4 5 6 7")
        print("    a b c d e f g h")


# главный цикл 

def parse_pair(prompt):
    line = input(prompt).strip().replace(',', ' ')
    parts = line.split()
    if len(parts) < 2:
        raise ValueError
    return int(parts[0]), int(parts[1])


def main():
    board = Board(True)
    board.print_board()

    while board.active():
        side = 'белых' if board.turn == 'w' else 'чёрных'
        print(f"\n=== Ход {side} ===")
        try:
            fx, fy = parse_pair("Выбор фигуры (x,y): ")
        except (ValueError, IndexError):
            print("Неверный ввод.")
            continue

        fig = board.get(fx, fy)
        if fig is None or not fig.alive or fig.color != board.turn:
            print("Неверный выбор фигуры.")
            continue

        try:
            mx, my = parse_pair("Ход (x,y): ")
        except (ValueError, IndexError):
            print("Неверный ввод.")
            continue

        # фиксируем флаг до хода
        will_promote = isinstance(fig, Pawn) and (
            (fig.color == 'w' and my == 7) or
            (fig.color == 'b' and my == 0)
        )

        if not fig.move(mx, my):
            print("Нелегальный ход.")
            continue

        # превращение пешки
        if will_promote and fig.alive:
            try:
                choice = int(input(
                    "Повышение (0-Ферзь, 1-Слон, 2-Ладья, 3-Конь): ").strip()
                )
            except ValueError:
                choice = 0
            if not (0 <= choice < 4):
                choice = 0
            fig.promote(choice)
            letter = ['Q', 'B', 'R', 'N'][choice]
            board.move_log[-1] = board.move_log[-1].replace('=?', '=' + letter)

        print("Ход:", board.move_log[-1])
        board.print_board()
        print("Партия:", ' '.join(board.move_log))

    # завершение партии
    print("\n=== Партия окончена ===")
    print("Последний ход:", board.move_log[-1] if board.move_log else '—')
    print("Итоговая запись партии:")
    print(' '.join(board.move_log))

    print("\ngetFigures():")
    print(board.getFigures())


if __name__ == '__main__':
    main()
