// Main homepage JavaScript

document.addEventListener('DOMContentLoaded', () => {
    initGenreFilters();
    initSearch();
    initRandomButton();
    renderGames(GAMES_DATABASE);
});

// Initialize genre filter buttons
function initGenreFilters() {
    const filtersContainer = document.getElementById('genre-filters');
    if (!filtersContainer) return;

    // Add genre buttons dynamically
    ALL_GENRES.forEach(genre => {
        const btn = document.createElement('button');
        btn.className = 'genre-btn';
        btn.dataset.genre = genre.id;
        btn.textContent = genre.name;
        btn.addEventListener('click', () => filterByGenre(genre.id));
        filtersContainer.appendChild(btn);
    });
}

// Filter games by genre
function filterByGenre(genreId) {
    // Update active button
    document.querySelectorAll('.genre-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.genre === genreId);
    });

    // Filter and render games
    const games = getGamesByGenre(genreId);
    renderGames(games);
}

// Initialize search functionality
function initSearch() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;

    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            const query = e.target.value.trim();
            if (query.length === 0) {
                // Reset to show all, respecting current genre filter
                const activeGenre = document.querySelector('.genre-btn.active');
                const genreId = activeGenre ? activeGenre.dataset.genre : 'all';
                renderGames(getGamesByGenre(genreId));
            } else {
                const results = searchGames(query);
                renderGames(results);
            }
        }, 300);
    });
}

// Initialize random game button
function initRandomButton() {
    const randomBtn = document.getElementById('random-btn');
    if (!randomBtn) return;

    randomBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const game = getRandomGame();
        window.location.href = `play.html?game=${game.id}`;
    });
}

// Render games grid
function renderGames(games) {
    const grid = document.getElementById('games-grid');
    if (!grid) return;

    if (games.length === 0) {
        grid.innerHTML = `
            <div class="no-results">
                <p>No games found. Try a different search or filter.</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = games.map(game => `
        <article class="game-card" onclick="playGame('${game.id}')">
            <img
                class="game-card-image"
                src="${game.screenshot}"
                alt="${game.title}"
                loading="lazy"
                onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 400 300%22><rect fill=%22%231a1a25%22 width=%22400%22 height=%22300%22/><text x=%2250%%22 y=%2250%%22 fill=%22%236b7280%22 text-anchor=%22middle%22 dy=%22.3em%22 font-family=%22system-ui%22>No Image</text></svg>'"
            >
            <div class="game-card-content">
                <h3 class="game-card-title">${game.title}</h3>
                <p class="game-card-year">${game.year}</p>
                <p class="game-card-description">${game.description}</p>
                <div class="game-card-genres">
                    ${game.genres.map(g => `<span class="genre-tag">${g}</span>`).join('')}
                </div>
            </div>
        </article>
    `).join('');
}

// Navigate to play page
function playGame(gameId) {
    window.location.href = `play.html?game=${gameId}`;
}
