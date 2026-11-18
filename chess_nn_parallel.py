import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import numpy as np
import multiprocessing as mp
from collections import Counter

from chess import Board, Pawn, Knight, Bishop, Rook, Queen, King



# макропараметры

COORD_DIM      = 64
COLOR_DIM      = 1
PAWN_STATE_DIM = 16
INPUT_DIM      = COORD_DIM + COLOR_DIM + PAWN_STATE_DIM   # = 81

HIDDEN_DIM = 128
NUM_HIDDEN = 10
OUTPUT_DIM = 64

POPULATION_SIZE         = 500
SURVIVORS               = 25
GENERATIONS             = 100
MUTATION_PARAMS         = 16
MUTATION_LAYERS         = 3
MAX_CROSSOVERS_PER_PAIR = 5
MAX_PLY                 = 100


# нормализация выхода
_OUTPUT_MULT = np.array(
    [16.0 if i % 2 == 0 else 2.0 for i in range(OUTPUT_DIM)],
    dtype=np.float32,
)

_tf_module = None

def _get_tf():
    global _tf_module
    if _tf_module is None:
        import tensorflow as _tf
        _tf_module = _tf
    return _tf_module


# модель
class EvoModel:
    def __init__(self):
        tf = _get_tf()
        self.Ws, self.bs = [], []
        for i in range(NUM_HIDDEN):
            in_dim = INPUT_DIM if i == 0 else HIDDEN_DIM
            W_init = (np.random.randn(in_dim, HIDDEN_DIM)
                      * np.sqrt(2.0 / in_dim)).astype(np.float32)
            self.Ws.append(tf.Variable(W_init))
            self.bs.append(tf.Variable(np.zeros(HIDDEN_DIM, dtype=np.float32)))
        Wo_init = (np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
                   * np.sqrt(2.0 / HIDDEN_DIM)).astype(np.float32)
        self.Wo = tf.Variable(Wo_init)
        self.bo = tf.Variable(np.zeros(OUTPUT_DIM, dtype=np.float32))
        self.score = -np.inf

    def forward(self, x):
        tf = _get_tf()
        h = tf.convert_to_tensor(x, dtype=tf.float32)
        for W, b in zip(self.Ws, self.bs):
            h = tf.nn.relu(tf.matmul(h, W) + b)
        raw = tf.matmul(h, self.Wo) + self.bo
        return tf.sigmoid(raw) * tf.constant(_OUTPUT_MULT)

    def copy(self):
        tf = _get_tf()
        m = EvoModel.__new__(EvoModel)
        m.Ws = [tf.Variable(W.numpy()) for W in self.Ws]
        m.bs = [tf.Variable(b.numpy()) for b in self.bs]
        m.Wo = tf.Variable(self.Wo.numpy())
        m.bo = tf.Variable(self.bo.numpy())
        m.score = self.score
        return m

    # (де)сериализация для передачи в воркеры
    def to_weights(self):
        return {
            'Ws': [w.numpy() for w in self.Ws],
            'bs': [b.numpy() for b in self.bs],
            'Wo': self.Wo.numpy(),
            'bo': self.bo.numpy(),
        }


# перевод форварда на чистый numpy для ускорения быстродействия
def numpy_forward(weights, x):
    h = np.asarray(x, dtype=np.float32)
    for W, b in zip(weights['Ws'], weights['bs']):
        h = h @ W + b
        np.maximum(h, 0.0, out=h)
    raw = h @ weights['Wo'] + weights['bo']
    return (1.0 / (1.0 + np.exp(-raw))) * _OUTPUT_MULT



# кросинговер / мутация
def crossover(a: EvoModel, b: EvoModel):
    ca, cb = a.copy(), b.copy()
    n = np.random.randint(1, MAX_CROSSOVERS_PER_PAIR + 1)
    n = min(n, NUM_HIDDEN)
    layers = np.random.choice(NUM_HIDDEN, size=n, replace=False)
    for li in layers:
        ca.Ws[li], cb.Ws[li] = cb.Ws[li], ca.Ws[li]
        ca.bs[li], cb.bs[li] = cb.bs[li], ca.bs[li]
    return ca, cb


def mutate(model: EvoModel):
    layers = np.random.choice(NUM_HIDDEN, size=MUTATION_LAYERS, replace=False)
    for li in layers:
        W = model.Ws[li]; b = model.bs[li]
        W_np = W.numpy().copy(); b_np = b.numpy().copy()
        W_flat = W_np.reshape(-1)
        total = W_flat.size + b_np.size
        idxs = np.random.choice(total, size=MUTATION_PARAMS, replace=False)
        for idx in idxs:
            val = np.float32(np.random.randn()) * np.float32(0.1)
            if idx < W_flat.size:
                W_flat[idx] = val
            else:
                b_np[idx - W_flat.size] = val
        W.assign(W_np); b.assign(b_np)


# оценка фигур / декодирование / вход
def piece_value(piece):
    if isinstance(piece, Pawn):              return 1
    if isinstance(piece, (Bishop, Knight)): return 3
    if isinstance(piece, Rook):             return 5
    if isinstance(piece, Queen):            return 8
    if isinstance(piece, King):             return 25
    return 0


def decode_move(piece, m_val, board):
    if m_val < 0.0:  m_val = 0.0
    if m_val >= 2.0: m_val = 2.0 - 1e-9
    x, y  = piece.x, piece.y
    color = piece.color
    fwd   = 1 if color == 'w' else -1
    rx    = 1 if color == 'w' else -1

    if isinstance(piece, Pawn):
        if m_val < 0.5:      dx, dy = -rx, fwd
        elif m_val < 1.0:    dx, dy =  rx, fwd
        elif m_val < 1.5:    dx, dy =   0, fwd
        else:                dx, dy =   0, 2 * fwd
        tx, ty = x + dx, y + dy
        return (tx, ty) if 0 <= tx < 8 and 0 <= ty < 8 else None

    if isinstance(piece, Knight):
        idx = min(int(m_val * 4), 7)
        offsets = [
            ( 2*rx,   fwd),   (  rx,   2*fwd),
            ( -rx,   2*fwd),  (-2*rx,   fwd),
            (-2*rx,  -fwd),   ( -rx,  -2*fwd),
            (  rx,  -2*fwd),  ( 2*rx,  -fwd),
        ]
        dx, dy = offsets[idx]
        tx, ty = x + dx, y + dy
        return (tx, ty) if 0 <= tx < 8 and 0 <= ty < 8 else None

    if isinstance(piece, (Bishop, Rook, Queen)):
        num_zones = 8 if isinstance(piece, Queen) else 4
        zone_size = 2.0 / num_zones
        zone_idx  = min(int(m_val / zone_size), num_zones - 1)
        a = zone_idx * zone_size
        b = a + zone_size
        dist = int(((m_val - a) / (b - a)) * 8)
        if dist == 0:
            return None
        all_dirs = [
            (   0,  fwd), (  rx,  fwd), (  rx,    0), (  rx, -fwd),
            (   0, -fwd), (-rx, -fwd), (-rx,    0), (-rx,  fwd),
        ]
        if isinstance(piece, Rook):
            dirs = [all_dirs[0], all_dirs[2], all_dirs[4], all_dirs[6]]
        elif isinstance(piece, Bishop):
            dirs = [all_dirs[1], all_dirs[3], all_dirs[5], all_dirs[7]]
        else:
            dirs = all_dirs
        ddx, ddy = dirs[zone_idx]
        tx, ty = x + ddx * dist, y + ddy * dist
        return (tx, ty) if 0 <= tx < 8 and 0 <= ty < 8 else None

    if isinstance(piece, King):
        zone_size = 0.2
        zone_idx  = min(int(m_val / zone_size), 9)
        if zone_idx == 8: return ('castle', True)
        if zone_idx == 9: return ('castle', False)
        all_dirs = [
            (   0,  fwd), (  rx,  fwd), (  rx,    0), (  rx, -fwd),
            (   0, -fwd), (-rx, -fwd), (-rx,    0), (-rx,  fwd),
        ]
        ddx, ddy = all_dirs[zone_idx]
        tx, ty = x + ddx, y + ddy
        return (tx, ty) if 0 <= tx < 8 and 0 <= ty < 8 else None

    return None


def build_input(board, color):
    figures = board.getFigures()
    inp = np.zeros(INPUT_DIM, dtype=np.float32)
    for i in range(0, 64, 2):
        xv, yv = figures[i], figures[i + 1]
        if xv < 0:
            inp[i] = inp[i + 1] = -1.0
        else:
            inp[i]     = xv / 7.0
            inp[i + 1] = yv / 7.0
    inp[64] = 1.0 if color == 'w' else -1.0
    inp[65:81] = board.getPawnStates()
    return inp, figures


# одна партия между моделями, принимает веса чтобы не зависеть от TF
def play_game(weights_white, weights_black, max_ply=MAX_PLY):
    board = Board(verbose=False)
    scores = {'w': 0.0, 'b': 0.0}
    end_reason = None
    ply = 0

    while board.active() and ply < max_ply:
        color = board.turn
        opp   = 'b' if color == 'w' else 'w'
        weights = weights_white if color == 'w' else weights_black

        inp, figures = build_input(board, color)
        out = numpy_forward(weights, inp)

        off    = 0 if color == 'w' else 32
        coords = figures[off:off + 32]

        move_made     = False
        illegal_count = 0

        for i in range(32):
            p_val = float(out[2 * i])
            p_idx = min(max(int(p_val), 0), 15)
            m_val = float(out[2 * i + 1])

            px = int(coords[2 * p_idx])
            py = int(coords[2 * p_idx + 1])
            if px < 0:
                illegal_count += 1
                continue

            piece = board.get(px, py)
            if piece is None or piece.color != color:
                illegal_count += 1
                continue

            target = decode_move(piece, m_val, board)
            if target is None:
                illegal_count += 1
                continue

            if isinstance(target, tuple) and target[0] == 'castle':
                if not piece.castle(target[1]):
                    illegal_count += 1
                    continue
                move_made = True
                scores[color] += 2.0
                break

            tx, ty = target
            captured = board.get(tx, ty)
            if isinstance(piece, Pawn) and captured is None \
                    and board.en_passant_target == (tx, ty):
                cy = ty - 1 if color == 'w' else ty + 1
                captured = board.get(tx, cy)

            if not piece.move(tx, ty):
                illegal_count += 1
                continue

            move_made = True
            scores[color] += 0.02
            if captured is not None and captured.color != color:
                scores[color] += piece_value(captured)

            if isinstance(piece, Pawn) and piece.alive and (
                (color == 'w' and piece.y == 7) or
                (color == 'b' and piece.y == 0)
            ):
                zone_idx = min(int(m_val / 0.5), 3)
                a = zone_idx * 0.5
                b = a + 0.5
                promo_idx = int(((m_val - a) / (b - a)) * 4)
                promo_idx = min(max(promo_idx, 0), 3)
                piece.promote(promo_idx)
                letter = ['Q', 'B', 'R', 'N'][promo_idx]
                board.move_log[-1] = board.move_log[-1].replace(
                    '=?', '=' + letter)
            break

        scores[color] -= 0.15 * illegal_count

        if not move_made:
            if not board._has_any_legal_move(color):
                king = board.find_king(color)
                if king is None or board.is_square_attacked(king.x, king.y, opp):
                    scores[opp] += 10
                    scores[color] = 0
                    end_reason = 'mate'
                else:
                    scores['w'] += 5
                    scores['b'] += 5
                    end_reason = 'stalemate'
            else:
                scores[color] = -10
                end_reason = 'premature'
            break

        ply += 1

    if end_reason is None:
        if not board.active():
            loser  = board.turn
            winner = 'b' if loser == 'w' else 'w'
            scores[loser] = 0
            end_reason = 'mate'
        elif ply >= max_ply:
            scores['w'] += 5
            scores['b'] += 5
            end_reason = 'stalemate'

    return scores['w'], scores['b'], list(board.move_log)



# воркер играет одну партию по индексам i, j из популяции.
def _play_worker(args):
    i, j, wa, wb, swap = args
    if swap:
        # wa играет чёрными, wb — белыми
        sw, sb, log = play_game(wb, wa)
        return i, j, sb, sw, log
    else:
        sw, sb, log = play_game(wa, wb)
        return i, j, sw, sb, log


# собственно эволюция
def evolve(seed=None, verbose=True, n_workers=None):
    if seed is not None:
        np.random.seed(seed)
        _get_tf().random.set_seed(seed)

    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)

    if verbose:
        print(f"Параллелизация: {n_workers} воркер(ов)")

    population = [EvoModel() for _ in range(POPULATION_SIZE)]
    history = []
    best_game = None

    ctx = mp.get_context('spawn')
    pool = None if n_workers == 1 else ctx.Pool(processes=n_workers)

    try:
        for gen in range(GENERATIONS):
            order = np.random.permutation(POPULATION_SIZE)
            scores = np.zeros(POPULATION_SIZE, dtype=np.float64)
            is_last = (gen == GENERATIONS - 1)

            tasks = []
            for k in range(POPULATION_SIZE):
                for m in range(5):
                    m_=m
                    if k+m>=POPULATION_SIZE:
                        m_ = m-10
                    i, j = int(order[k]), int(order[k+m_])
                    wa = population[i].to_weights()
                    wb = population[j].to_weights()
                    swap = bool(np.random.rand() < 0.5)
                    tasks.append((i, j, wa, wb, swap))

            # запуск воркеров
            if pool is None:
                results = [_play_worker(t) for t in tasks]
            else:
                results = pool.map(_play_worker, tasks, chunksize=1)

            last_gen_games = []
            for (i, j, si, sj, log) in results:
                scores[i] += si
                scores[j] += sj
                if is_last:
                    last_gen_games.append((si + sj, log))

            if is_last and last_gen_games:
                best_game = max(last_gen_games, key=lambda t: t[0])

            for i, m in enumerate(population):
                m.score = float(scores[i])

            # отбор
            order_idx = np.argsort(-scores)
            survivors = [population[i].copy() for i in order_idx[:SURVIVORS]]
            best = float(scores[order_idx[0]])
            mean = float(scores.mean())
            history.append(best)

            if verbose:
                print(f"gen {gen+1:3d}/{GENERATIONS} | "
                      f"best = {best:9.3f} | mean = {mean:9.3f}")

            # новое поколение
            new_pop = [s.copy() for s in survivors]
            pair_counter = Counter()
            while len(new_pop) < POPULATION_SIZE:
                i, j = np.random.choice(SURVIVORS, size=2, replace=False)
                i, j = int(i), int(j)
                key = (min(i, j), max(i, j))
                if pair_counter[key] < MAX_CROSSOVERS_PER_PAIR:
                    ca, cb = crossover(survivors[i], survivors[j])
                    pair_counter[key] += 1
                else:
                    ca = survivors[i].copy()
                    cb = survivors[j].copy()
                mutate(ca)
                mutate(cb)
                new_pop.append(ca)
                if len(new_pop) < POPULATION_SIZE:
                    new_pop.append(cb)
            population = new_pop[:POPULATION_SIZE]
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return population, history, best_game



# Mann vs Machine
def _parse_pair(prompt):
    line = input(prompt).strip().replace(',', ' ')
    parts = line.split()
    if len(parts) < 2:
        raise ValueError
    return int(parts[0]), int(parts[1])


def play_human_vs_model(weights, human_color='w', max_ply=MAX_PLY):
    board = Board(verbose=True)
    board.print_board()

    while board.active() and len(board.move_log) < max_ply:
        color = board.turn
        side_name = 'белых' if color == 'w' else 'чёрных'

        if color == human_color:
            print(f"\n=== Ваш ход ({side_name}) ===")
            try:
                fx, fy = _parse_pair("Выбор фигуры (x,y): ")
                fig = board.get(fx, fy)
                if fig is None or not fig.alive or fig.color != color:
                    print("Неверный выбор фигуры.")
                    continue
                mx, my = _parse_pair("Ход (x,y): ")
            except (ValueError, IndexError):
                print("Неверный ввод.")
                continue

            will_promote = isinstance(fig, Pawn) and (
                (color == 'w' and my == 7) or (color == 'b' and my == 0))
            if not fig.move(mx, my):
                print("Нелегальный ход.")
                continue

            if will_promote and fig.alive:
                try:
                    choice = int(input(
                        "Повышение (0-Ферзь, 1-Слон, 2-Ладья, 3-Конь): "
                    ).strip())
                except ValueError:
                    choice = 0
                if not (0 <= choice < 4):
                    choice = 0
                fig.promote(choice)
                board.move_log[-1] = board.move_log[-1].replace(
                    '=?', '=' + ['Q', 'B', 'R', 'N'][choice])

        else:
            print(f"\n=== Ход модели ({side_name}) ===")
            inp, figures = build_input(board, color)
            out = numpy_forward(weights, inp)
            off = 0 if color == 'w' else 32
            coords = figures[off:off + 32]

            move_made = False
            for i in range(32):
                p_val = float(out[2 * i])
                p_idx = min(max(int(p_val), 0), 15)
                m_val = float(out[2 * i + 1])
                px = int(coords[2 * p_idx]); py = int(coords[2 * p_idx + 1])
                if px < 0:
                    continue
                piece = board.get(px, py)
                if piece is None or piece.color != color:
                    continue
                target = decode_move(piece, m_val, board)
                if target is None:
                    continue
                if isinstance(target, tuple) and target[0] == 'castle':
                    if not piece.castle(target[1]):
                        continue
                    move_made = True
                    break
                tx, ty = target
                if not piece.move(tx, ty):
                    continue
                move_made = True
                if isinstance(piece, Pawn) and piece.alive and (
                    (color == 'w' and piece.y == 7) or
                    (color == 'b' and piece.y == 0)
                ):
                    zone_idx = min(int(m_val / 0.5), 3)
                    a = zone_idx * 0.5; b = a + 0.5
                    promo_idx = int(((m_val - a) / (b - a)) * 4)
                    promo_idx = min(max(promo_idx, 0), 3)
                    piece.promote(promo_idx)
                    board.move_log[-1] = board.move_log[-1].replace(
                        '=?', '=' + ['Q', 'B', 'R', 'N'][promo_idx])
                break

            if not move_made:
                print("Модель не нашла легальный ход.")
                break

        print("Ход:", board.move_log[-1])
        board.print_board()
        print("Партия:", ' '.join(board.move_log))

    print("\n=== Партия окончена ===")
    print(' '.join(board.move_log))



# сохранение / загрузка модели
def save_model(model: EvoModel, path='best_chess_model.npz'):
    w = model.to_weights()
    np.savez(
        path,
        Wo=w['Wo'], bo=w['bo'],
        **{f'W{i}': a for i, a in enumerate(w['Ws'])},
        **{f'b{i}': a for i, a in enumerate(w['bs'])},
    )
    return path


def load_model(path='best_chess_model.npz') -> EvoModel:
    data = np.load(path)
    tf = _get_tf()
    m = EvoModel.__new__(EvoModel)
    m.Ws, m.bs = [], []
    for i in range(NUM_HIDDEN):
        m.Ws.append(tf.Variable(data[f'W{i}'].astype(np.float32)))
        m.bs.append(tf.Variable(data[f'b{i}'].astype(np.float32)))
    m.Wo = tf.Variable(data['Wo'].astype(np.float32))
    m.bo = tf.Variable(data['bo'].astype(np.float32))
    m.score = 0.0
    if m.Ws[0].shape[0] != INPUT_DIM:
        raise ValueError(
            f"Ожидалось {INPUT_DIM} входов, в файле {m.Ws[0].shape[0]}")
    if m.Wo.shape[1] != OUTPUT_DIM:
        raise ValueError(
            f"Ожидалось {OUTPUT_DIM} выходов, в файле {m.Wo.shape[1]}")
    return m


# меню
def _ask(prompt, default=''):
    try:
        s = input(prompt).strip()
        return s if s else default
    except EOFError:
        return default


def _print_best_game(best_game):
    if best_game is None:
        return
    total, log = best_game
    print("\n" + "=" * 60)
    print(f"ЛУЧШАЯ ПАРТИЯ ИТОГОВОГО ПОКОЛЕНИЯ "
          f"(суммарный score = {total:.3f})")
    print("=" * 60)
    for i in range(0, len(log), 2):
        num = i // 2 + 1
        w_move = log[i]
        b_move = log[i + 1] if (i + 1) < len(log) else ''
        print(f"{num:3d}. {w_move:<9s} {b_move}")
    print(' '.join(log))
    print("=" * 60)


def _play_against(model_or_weights):
    if isinstance(model_or_weights, EvoModel):
        weights = model_or_weights.to_weights()
    else:
        weights = model_or_weights
    choice = _ask("Ваш цвет (w = белые, b = чёрные, Enter = белые): ",
                  default='w').lower()
    play_human_vs_model(weights, human_color='b' if choice == 'b' else 'w')


def main_menu():
    print("=" * 60)
    print("ШАХМАТНАЯ НЕЙРОСЕТЬ")
    print("=" * 60)
    print("  1 — обучить новую модель")
    print("  2 — загрузить best_chess_model.npz и сыграть")
    print("  3 — загрузить модель по пути и сыграть")
    choice = _ask("Выбор (Enter = 1): ", default='1')

    if choice == '2':
        try:
            model = load_model('best_chess_model.npz')
        except FileNotFoundError:
            print("Файл не найден.")
            return
        _play_against(model)
        return

    if choice == '3':
        path = _ask("Путь к файлу модели: ")
        if not path:
            return
        try:
            model = load_model(path)
        except (FileNotFoundError, ValueError) as e:
            print("Ошибка загрузки:", e)
            return
        _play_against(model)
        return

    # обучение
    n_str = _ask(f"Число воркеров (Enter = {os.cpu_count()}): ",
                 default=str(os.cpu_count()))
    try:
        n_workers = max(1, int(n_str))
    except ValueError:
        n_workers = max(1, os.cpu_count())

    pop, hist, best_game = evolve(seed=42, verbose=True, n_workers=n_workers)
    print("\nОбучение завершено.")
    print("Лучший score последнего поколения:", hist[-1])

    best = max(pop, key=lambda m: m.score)
    save_model(best, 'best_chess_model.npz')
    print("Модель сохранена в best_chess_model.npz")

    _print_best_game(best_game)
    _play_against(best)


if __name__ == '__main__':
    main_menu()
