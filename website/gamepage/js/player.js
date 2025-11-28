// Game Player JavaScript - js-dos integration

let currentGame = null;
let dosInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    // Get game ID from URL
    const params = new URLSearchParams(window.location.search);
    const gameId = params.get('game');

    if (!gameId) {
        showError('No game specified. Please select a game from the homepage.');
        return;
    }

    currentGame = getGameById(gameId);
    if (!currentGame) {
        showError('Game not found. Please select a game from the homepage.');
        return;
    }

    // Update page info
    updateGameInfo();
    initControls();
    initResizable();
    loadGame();
});

// Update game information on page
function updateGameInfo() {
    document.title = `${currentGame.title} - RetroVault`;
    document.getElementById('game-title').textContent = currentGame.title;
    document.getElementById('game-description').textContent = currentGame.description;

    const genresContainer = document.getElementById('game-genres');
    genresContainer.innerHTML = currentGame.genres
        .map(g => `<span class="genre-tag">${g}</span>`)
        .join('');
}

// Initialize control buttons
function initControls() {
    // Fullscreen button
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    fullscreenBtn.addEventListener('click', toggleFullscreen);

    // Settings button
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const closeSettings = document.getElementById('close-settings');

    settingsBtn.addEventListener('click', () => {
        settingsModal.classList.remove('hidden');
    });

    closeSettings.addEventListener('click', () => {
        settingsModal.classList.add('hidden');
    });

    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.add('hidden');
        }
    });

    // Size presets
    document.querySelectorAll('.size-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const width = parseInt(btn.dataset.width);
            const height = parseInt(btn.dataset.height);
            resizeGameWindow(width, height);

            // Update active state
            document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Scale slider
    const scaleSlider = document.getElementById('scale-slider');
    const scaleValue = document.getElementById('scale-value');
    scaleSlider.addEventListener('input', () => {
        const scale = scaleSlider.value;
        scaleValue.textContent = `${scale}%`;
        applyScale(scale / 100);
    });

    // Random game button
    const randomBtn = document.getElementById('random-btn');
    if (randomBtn) {
        randomBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const game = getRandomGame();
            window.location.href = `play.html?game=${game.id}`;
        });
    }
}

// Toggle fullscreen
function toggleFullscreen() {
    const wrapper = document.getElementById('game-wrapper');

    if (!document.fullscreenElement) {
        if (wrapper.requestFullscreen) {
            wrapper.requestFullscreen();
        } else if (wrapper.webkitRequestFullscreen) {
            wrapper.webkitRequestFullscreen();
        } else if (wrapper.msRequestFullscreen) {
            wrapper.msRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        }
    }
}

// Initialize resizable game window
function initResizable() {
    const wrapper = document.getElementById('game-wrapper');
    const handle = document.getElementById('resize-handle');

    let isResizing = false;
    let startX, startY, startWidth, startHeight;

    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startY = e.clientY;
        startWidth = wrapper.offsetWidth;
        startHeight = wrapper.offsetHeight;

        document.addEventListener('mousemove', resize);
        document.addEventListener('mouseup', stopResize);
        e.preventDefault();
    });

    function resize(e) {
        if (!isResizing) return;

        const width = startWidth + (e.clientX - startX);
        const height = startHeight + (e.clientY - startY);

        // Maintain 4:3 aspect ratio (optional)
        // const constrainedHeight = width * 0.75;

        wrapper.style.width = Math.max(320, width) + 'px';
        wrapper.style.height = Math.max(240, height) + 'px';
    }

    function stopResize() {
        isResizing = false;
        document.removeEventListener('mousemove', resize);
        document.removeEventListener('mouseup', stopResize);
    }
}

// Resize game window to specific dimensions
function resizeGameWindow(width, height) {
    const wrapper = document.getElementById('game-wrapper');
    wrapper.style.width = width + 'px';
    wrapper.style.height = height + 'px';
}

// Apply scale transform
function applyScale(scale) {
    const wrapper = document.getElementById('game-wrapper');
    wrapper.style.transform = `scale(${scale})`;
    wrapper.style.transformOrigin = 'top center';
}

// Load game using js-dos
async function loadGame() {
    const container = document.getElementById('dos-container');

    // Show loading state
    container.innerHTML = `
        <div class="loading-overlay">
            <div class="loading-spinner"></div>
            <p class="loading-text">Loading ${currentGame.title}...</p>
        </div>
    `;

    try {
        // Construct the Internet Archive embed URL
        const archiveUrl = `https://archive.org/embed/${currentGame.archiveId}`;

        // Create an iframe to embed the Internet Archive's player
        // This is the simplest approach - uses Archive's own emulation
        container.innerHTML = `
            <iframe
                src="${archiveUrl}"
                style="width: 100%; height: 100%; border: none;"
                allowfullscreen
                allow="autoplay; fullscreen"
            ></iframe>
        `;

        // Alternative: Direct js-dos implementation
        // Uncomment below for self-hosted games with js-dos
        /*
        if (typeof Dos !== 'undefined') {
            const bundleUrl = `games/${currentGame.id}.jsdos`;

            dosInstance = await Dos(container, {
                url: bundleUrl,
                autoStart: true
            });

            // Apply saved controller mappings
            applyControllerMappings();
        }
        */

    } catch (error) {
        console.error('Error loading game:', error);
        showError('Failed to load game. Please try again later.');
    }
}

// Show error state
function showError(message) {
    const container = document.getElementById('dos-container');
    container.innerHTML = `
        <div class="error-state">
            <div class="error-icon">!</div>
            <p class="error-message">${message}</p>
            <a href="index.html" class="btn btn-primary">Back to Games</a>
        </div>
    `;
}

// Apply controller mappings from localStorage
function applyControllerMappings() {
    const savedMappings = localStorage.getItem('controllerMappings');
    if (savedMappings && dosInstance) {
        const mappings = JSON.parse(savedMappings);
        // Apply mappings to js-dos instance
        // This would integrate with js-dos's input handling
        console.log('Controller mappings loaded:', mappings);
    }
}

// Gamepad polling for js-dos integration
let gamepadPollInterval = null;

function startGamepadPolling() {
    if (gamepadPollInterval) return;

    const settings = JSON.parse(localStorage.getItem('controllerSettings') || '{}');
    const mappings = JSON.parse(localStorage.getItem('controllerMappings') || '{}');
    const deadzone = settings.deadzone || 0.15;
    const analogToDpad = settings.analogToDpad !== false;

    gamepadPollInterval = setInterval(() => {
        const gamepads = navigator.getGamepads();

        for (const gamepad of gamepads) {
            if (!gamepad) continue;

            // Handle buttons
            gamepad.buttons.forEach((button, index) => {
                if (button.pressed && mappings[index]) {
                    simulateKeyPress(mappings[index]);
                }
            });

            // Handle analog sticks as arrow keys
            if (analogToDpad) {
                const leftX = gamepad.axes[0];
                const leftY = gamepad.axes[1];

                if (leftX < -deadzone) simulateKeyPress('ArrowLeft');
                if (leftX > deadzone) simulateKeyPress('ArrowRight');
                if (leftY < -deadzone) simulateKeyPress('ArrowUp');
                if (leftY > deadzone) simulateKeyPress('ArrowDown');
            }
        }
    }, 16); // ~60fps
}

function stopGamepadPolling() {
    if (gamepadPollInterval) {
        clearInterval(gamepadPollInterval);
        gamepadPollInterval = null;
    }
}

function simulateKeyPress(key) {
    // This would send key events to the js-dos instance
    // Implementation depends on js-dos API
    if (dosInstance && dosInstance.sendKeyEvent) {
        dosInstance.sendKeyEvent(key, true);
        setTimeout(() => {
            dosInstance.sendKeyEvent(key, false);
        }, 50);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopGamepadPolling();
    if (dosInstance && dosInstance.exit) {
        dosInstance.exit();
    }
});

// Start gamepad polling when gamepad connected
window.addEventListener('gamepadconnected', (e) => {
    console.log('Gamepad connected:', e.gamepad.id);
    startGamepadPolling();
});

window.addEventListener('gamepaddisconnected', () => {
    stopGamepadPolling();
});
