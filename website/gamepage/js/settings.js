// Controller Settings JavaScript

// Default key mappings
const DEFAULT_MAPPINGS = {
    0: 'Space',      // A - Jump/Action
    1: 'Escape',     // B - Back/Cancel
    2: 'KeyX',       // X - Secondary action
    3: 'KeyZ',       // Y - Tertiary action
    4: 'KeyQ',       // LB
    5: 'KeyE',       // RB
    6: 'ShiftLeft',  // LT
    7: 'ControlLeft', // RT
    8: 'Tab',        // Select
    9: 'Enter',      // Start
    12: 'ArrowUp',   // D-pad Up
    13: 'ArrowDown', // D-pad Down
    14: 'ArrowLeft', // D-pad Left
    15: 'ArrowRight' // D-pad Right
};

// DOOM preset
const DOOM_PRESET = {
    0: 'ControlLeft', // A - Fire
    1: 'Space',       // B - Use/Open
    2: 'KeyE',        // X - Strafe Right
    3: 'KeyQ',        // Y - Strafe Left
    4: 'Comma',       // LB - Prev weapon
    5: 'Period',      // RB - Next weapon
    6: 'ShiftLeft',   // LT - Run
    7: 'ControlLeft', // RT - Fire
    8: 'Tab',         // Select - Map
    9: 'Escape',      // Start - Menu
    12: 'ArrowUp',
    13: 'ArrowDown',
    14: 'ArrowLeft',
    15: 'ArrowRight'
};

// Platformer preset
const PLATFORMER_PRESET = {
    0: 'Space',       // A - Jump
    1: 'Escape',      // B - Back
    2: 'KeyX',        // X - Attack
    3: 'KeyC',        // Y - Special
    4: 'PageUp',      // LB
    5: 'PageDown',    // RB
    6: 'ShiftLeft',   // LT - Run
    7: 'ControlLeft', // RT
    8: 'Tab',         // Select
    9: 'Enter',       // Start
    12: 'ArrowUp',
    13: 'ArrowDown',
    14: 'ArrowLeft',
    15: 'ArrowRight'
};

let currentMappings = {};
let activeInput = null;
let gamepadPollInterval = null;
let previousButtonStates = {}; // Track button states to detect new presses

document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    initKeyInputs();
    initButtons();
    initAnalogSettings();
    startGamepadDetection();
    initRandomButton();
});

// Load saved settings
function loadSettings() {
    const saved = localStorage.getItem('controllerMappings');
    currentMappings = saved ? JSON.parse(saved) : { ...DEFAULT_MAPPINGS };

    // Update input fields
    Object.entries(currentMappings).forEach(([button, key]) => {
        const input = document.querySelector(`.key-input[data-button="${button}"]`);
        if (input) {
            input.value = formatKeyName(key);
        }
    });

    // Load analog settings
    const settings = JSON.parse(localStorage.getItem('controllerSettings') || '{}');

    const deadzoneSlider = document.getElementById('deadzone');
    const deadzoneValue = document.getElementById('deadzone-value');
    if (deadzoneSlider && settings.deadzone !== undefined) {
        deadzoneSlider.value = settings.deadzone;
        deadzoneValue.textContent = settings.deadzone.toFixed(2);
    }

    const analogToDpad = document.getElementById('analog-to-dpad');
    if (analogToDpad) {
        analogToDpad.checked = settings.analogToDpad !== false;
    }
}

// Initialize key input fields
function initKeyInputs() {
    document.querySelectorAll('.key-input').forEach(input => {
        input.addEventListener('click', () => {
            // Remove listening state from other inputs
            document.querySelectorAll('.key-input').forEach(i => {
                i.classList.remove('listening');
            });

            input.classList.add('listening');
            input.value = 'Press key or button...';
            activeInput = input;

            // Reset previous button states so we detect fresh presses
            previousButtonStates = {};
        });
    });

    // Listen for key presses
    document.addEventListener('keydown', (e) => {
        if (!activeInput) return;

        // Don't capture Escape - it cancels
        if (e.key === 'Escape') return;

        e.preventDefault();
        const button = activeInput.dataset.button;
        const keyCode = e.code;

        currentMappings[button] = keyCode;
        activeInput.value = formatKeyName(keyCode);
        activeInput.classList.remove('listening');
        activeInput = null;
    });

    // Cancel on escape
    document.addEventListener('keyup', (e) => {
        if (e.key === 'Escape' && activeInput) {
            const button = activeInput.dataset.button;
            activeInput.value = formatKeyName(currentMappings[button] || '');
            activeInput.classList.remove('listening');
            activeInput = null;
        }
    });
}

// Initialize buttons
function initButtons() {
    document.getElementById('save-mapping').addEventListener('click', saveSettings);
    document.getElementById('reset-mapping').addEventListener('click', resetToDefault);
    document.getElementById('load-preset-doom').addEventListener('click', () => loadPreset(DOOM_PRESET));
    document.getElementById('load-preset-platformer').addEventListener('click', () => loadPreset(PLATFORMER_PRESET));
}

// Initialize analog settings
function initAnalogSettings() {
    const deadzoneSlider = document.getElementById('deadzone');
    const deadzoneValue = document.getElementById('deadzone-value');

    deadzoneSlider.addEventListener('input', () => {
        deadzoneValue.textContent = parseFloat(deadzoneSlider.value).toFixed(2);
    });
}

// Save settings to localStorage
function saveSettings() {
    localStorage.setItem('controllerMappings', JSON.stringify(currentMappings));

    const settings = {
        deadzone: parseFloat(document.getElementById('deadzone').value),
        analogToDpad: document.getElementById('analog-to-dpad').checked
    };
    localStorage.setItem('controllerSettings', JSON.stringify(settings));

    showSaveIndicator();
}

// Reset to default mappings
function resetToDefault() {
    currentMappings = { ...DEFAULT_MAPPINGS };
    Object.entries(currentMappings).forEach(([button, key]) => {
        const input = document.querySelector(`.key-input[data-button="${button}"]`);
        if (input) {
            input.value = formatKeyName(key);
        }
    });
}

// Load a preset
function loadPreset(preset) {
    currentMappings = { ...preset };
    Object.entries(currentMappings).forEach(([button, key]) => {
        const input = document.querySelector(`.key-input[data-button="${button}"]`);
        if (input) {
            input.value = formatKeyName(key);
        }
    });
}

// Show save indicator
function showSaveIndicator() {
    const actions = document.querySelector('.mapping-actions');
    let indicator = document.querySelector('.save-indicator');

    if (!indicator) {
        indicator = document.createElement('span');
        indicator.className = 'save-indicator';
        indicator.textContent = 'Saved!';
        actions.appendChild(indicator);
    }

    indicator.classList.add('show');
    setTimeout(() => {
        indicator.classList.remove('show');
    }, 2000);
}

// Format key name for display
function formatKeyName(code) {
    if (!code) return '';

    const keyMap = {
        'Space': 'Space',
        'Escape': 'Esc',
        'Enter': 'Enter',
        'Tab': 'Tab',
        'ShiftLeft': 'L-Shift',
        'ShiftRight': 'R-Shift',
        'ControlLeft': 'L-Ctrl',
        'ControlRight': 'R-Ctrl',
        'AltLeft': 'L-Alt',
        'AltRight': 'R-Alt',
        'ArrowUp': 'Up',
        'ArrowDown': 'Down',
        'ArrowLeft': 'Left',
        'ArrowRight': 'Right',
        'Backspace': 'Backspace',
        'Delete': 'Delete',
        'PageUp': 'PgUp',
        'PageDown': 'PgDn',
        'Home': 'Home',
        'End': 'End',
        'Comma': ',',
        'Period': '.'
    };

    if (keyMap[code]) return keyMap[code];
    if (code.startsWith('Key')) return code.substring(3);
    if (code.startsWith('Digit')) return code.substring(5);
    if (code.startsWith('Numpad')) return 'Num' + code.substring(6);

    return code;
}

// Start gamepad detection and visualization
function startGamepadDetection() {
    window.addEventListener('gamepadconnected', (e) => {
        console.log('Gamepad connected:', e.gamepad.id);
        updateControllerInfo(e.gamepad);
        startGamepadPolling();
    });

    window.addEventListener('gamepaddisconnected', () => {
        console.log('Gamepad disconnected');
        updateControllerInfo(null);
        stopGamepadPolling();
    });

    // Check for already-connected gamepads
    const gamepads = navigator.getGamepads();
    for (const gamepad of gamepads) {
        if (gamepad) {
            updateControllerInfo(gamepad);
            startGamepadPolling();
            break;
        }
    }

    // Also start polling immediately to detect gamepads that need a button press
    // Some browsers require user interaction before reporting gamepads
    startGamepadPolling();
}

// Update controller info display
function updateControllerInfo(gamepad) {
    const infoContainer = document.getElementById('controller-info');

    if (!gamepad) {
        infoContainer.innerHTML = `
            <p class="no-controller">No controller detected. Connect a controller and press any button.</p>
        `;
        infoContainer.classList.remove('connected');
        return;
    }

    infoContainer.innerHTML = `
        <p class="controller-name">${gamepad.id}</p>
        <p class="controller-id">${gamepad.buttons.length} buttons, ${gamepad.axes.length} axes</p>
    `;
    infoContainer.classList.add('connected');

    // Initialize button display
    initButtonDisplay(gamepad.buttons.length);
}

// Initialize button indicators
function initButtonDisplay(buttonCount) {
    const display = document.getElementById('buttons-display');
    display.innerHTML = '';

    for (let i = 0; i < Math.min(buttonCount, 16); i++) {
        const btn = document.createElement('div');
        btn.className = 'button-indicator';
        btn.id = `btn-${i}`;
        btn.textContent = i;
        display.appendChild(btn);
    }
}

// Start polling gamepad for visual feedback and button assignment
let lastKnownGamepad = null;

function startGamepadPolling() {
    if (gamepadPollInterval) return;

    gamepadPollInterval = setInterval(() => {
        const gamepads = navigator.getGamepads();
        let foundGamepad = null;

        for (const gamepad of gamepads) {
            if (!gamepad) continue;
            foundGamepad = gamepad;

            // Detect newly connected gamepad
            if (!lastKnownGamepad) {
                updateControllerInfo(gamepad);
            }

            // Update button indicators and check for assignment
            gamepad.buttons.forEach((button, index) => {
                const indicator = document.getElementById(`btn-${index}`);
                if (indicator) {
                    indicator.classList.toggle('pressed', button.pressed);
                }

                // Check for new button press for assignment
                const wasPressed = previousButtonStates[index] || false;
                const isPressed = button.pressed;

                // Detect new press (wasn't pressed, now is pressed)
                if (isPressed && !wasPressed) {
                    // If no input is active, auto-select the matching row
                    // If input is active, this lets user know which button they pressed
                    assignGamepadButton(index);
                }

                previousButtonStates[index] = isPressed;
            });

            // Update stick indicators
            const leftStick = document.getElementById('left-stick');
            const rightStick = document.getElementById('right-stick');

            if (leftStick && gamepad.axes.length >= 2) {
                const x = gamepad.axes[0] * 20;
                const y = gamepad.axes[1] * 20;
                leftStick.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
            }

            if (rightStick && gamepad.axes.length >= 4) {
                const x = gamepad.axes[2] * 20;
                const y = gamepad.axes[3] * 20;
                rightStick.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
            }

            break; // Only process first gamepad
        }

        lastKnownGamepad = foundGamepad;
    }, 16); // ~60fps
}

// Assign a gamepad button - auto-select the matching row when gamepad button pressed
function assignGamepadButton(gamepadButtonIndex) {
    // Find the input field for this gamepad button
    const targetInput = document.querySelector(`.key-input[data-button="${gamepadButtonIndex}"]`);

    if (targetInput) {
        // If we're already on a different input, switch to this one
        if (activeInput && activeInput !== targetInput) {
            activeInput.classList.remove('listening');
            const oldButton = activeInput.dataset.button;
            activeInput.value = formatKeyName(currentMappings[oldButton] || '');
        }

        // Activate this input
        document.querySelectorAll('.key-input').forEach(i => i.classList.remove('listening'));
        targetInput.classList.add('listening');
        targetInput.value = `${getGamepadButtonName(gamepadButtonIndex)} - press a key`;
        activeInput = targetInput;

        // Scroll to make it visible
        targetInput.scrollIntoView({ behavior: 'smooth', block: 'center' });

        // Reset button states to avoid re-triggering
        previousButtonStates = {};
    }
}

// Get friendly name for gamepad button
function getGamepadButtonName(index) {
    const names = {
        0: 'A',
        1: 'B',
        2: 'X',
        3: 'Y',
        4: 'LB',
        5: 'RB',
        6: 'LT',
        7: 'RT',
        8: 'Select',
        9: 'Start',
        10: 'L3',
        11: 'R3',
        12: 'D-Up',
        13: 'D-Down',
        14: 'D-Left',
        15: 'D-Right'
    };
    return names[index] || `Button ${index}`;
}

// Stop gamepad polling
function stopGamepadPolling() {
    if (gamepadPollInterval) {
        clearInterval(gamepadPollInterval);
        gamepadPollInterval = null;
    }
}

// Random game button
function initRandomButton() {
    const randomBtn = document.getElementById('random-btn');
    if (randomBtn) {
        randomBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const game = getRandomGame();
            window.location.href = `play.html?game=${game.id}`;
        });
    }
}
