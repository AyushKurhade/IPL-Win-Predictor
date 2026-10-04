/**
 * IPL Match Prediction - Frontend JavaScript
 * Handles input validation, fetch() POST to /predict, and dynamic probability rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictionForm');
  const predictBtn = document.getElementById('predictBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const errorAlert = document.getElementById('errorAlert');
  const errorMessageText = document.getElementById('errorMessageText');
  const resultContainer = document.getElementById('resultContainer');

  // Result display elements
  const winnerNameDisplay = document.getElementById('winnerNameDisplay');
  const winnerProbDisplay = document.getElementById('winnerProbDisplay');
  const battingTeamNameDisplay = document.getElementById('battingTeamNameDisplay');
  const battingTeamProbDisplay = document.getElementById('battingTeamProbDisplay');
  const battingProgressBar = document.getElementById('battingProgressBar');
  const bowlingTeamNameDisplay = document.getElementById('bowlingTeamNameDisplay');
  const bowlingTeamProbDisplay = document.getElementById('bowlingTeamProbDisplay');
  const bowlingProgressBar = document.getElementById('bowlingProgressBar');

  // Match stats elements
  const statRunsLeft = document.getElementById('statRunsLeft');
  const statBallsLeft = document.getElementById('statBallsLeft');
  const statWicketsLeft = document.getElementById('statWicketsLeft');
  const statCRR = document.getElementById('statCRR');
  const statRRR = document.getElementById('statRRR');

  function showError(message) {
    errorMessageText.textContent = message;
    errorAlert.classList.remove('d-none');
    resultContainer.classList.add('d-none');
    errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function hideError() {
    errorAlert.classList.add('d-none');
  }

  function setLoading(isLoading) {
    if (isLoading) {
      predictBtn.disabled = true;
      btnSpinner.classList.remove('d-none');
      btnText.textContent = ' Calculating Prediction...';
    } else {
      predictBtn.disabled = false;
      btnSpinner.classList.add('d-none');
      btnText.textContent = 'Predict Winner';
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    // 1. Gather form values
    const battingTeam = document.getElementById('battingTeam').value;
    const bowlingTeam = document.getElementById('bowlingTeam').value;
    const hostCity = document.getElementById('hostCity').value;
    const targetScoreStr = document.getElementById('targetScore').value.trim();
    const currentScoreStr = document.getElementById('currentScore').value.trim();
    const oversCompletedStr = document.getElementById('oversCompleted').value.trim();
    const wicketsOutStr = document.getElementById('wicketsOut').value.trim();

    // 2. Client-side Validations
    if (!battingTeam) {
      showError('Please select a Batting Team.');
      return;
    }

    if (!bowlingTeam) {
      showError('Please select a Bowling Team.');
      return;
    }

    if (battingTeam === bowlingTeam) {
      showError('Please select different batting and bowling teams.');
      return;
    }

    if (!hostCity) {
      showError('Please select a Host City.');
      return;
    }

    if (targetScoreStr === '') {
      showError('Please enter the Target Score.');
      return;
    }

    if (currentScoreStr === '') {
      showError('Please enter the Current Score.');
      return;
    }

    if (oversCompletedStr === '') {
      showError('Please enter the Overs Completed.');
      return;
    }

    if (wicketsOutStr === '') {
      showError('Please enter the Wickets Out.');
      return;
    }

    const target = parseInt(targetScoreStr, 10);
    const currentScore = parseInt(currentScoreStr, 10);
    const overs = parseFloat(oversCompletedStr);
    const wickets = parseInt(wicketsOutStr, 10);

    if (isNaN(target) || isNaN(currentScore) || isNaN(overs) || isNaN(wickets)) {
      showError('Please enter valid numeric values for scores, overs, and wickets.');
      return;
    }

    if (target <= 0) {
      showError('Target Score must be greater than 0.');
      return;
    }

    if (currentScore < 0) {
      showError('Current Score cannot be negative.');
      return;
    }

    if (currentScore >= target) {
      showError(`Current score (${currentScore}) has reached or exceeded the target (${target}). Batting team has already won!`);
      return;
    }

    if (overs < 0 || overs > 20) {
      showError('Overs Completed must be between 0 and 20.');
      return;
    }

    // Verify ball portion of decimal over notation (0 to 5 balls per over)
    const overWhole = Math.floor(overs);
    const ballPart = Math.round((overs - overWhole) * 10);
    if (ballPart > 5) {
      showError(`Invalid overs format "${overs}". A cricket over has 6 balls (.0 to .5). Example: 12.3`);
      return;
    }

    if (wickets < 0 || wickets > 10) {
      showError('Wickets Out must be between 0 and 10.');
      return;
    }

    // 3. Prepare Payload
    const payload = {
      batting_team: battingTeam,
      bowling_team: bowlingTeam,
      city: hostCity,
      target: target,
      current_score: currentScore,
      overs: overs,
      wickets: wickets
    };

    // 4. Send Request via Fetch API
    setLoading(true);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        showError(result.error || 'Prediction failed. Please check inputs and try again.');
        return;
      }

      // 5. Render Result Data smoothly
      winnerNameDisplay.textContent = result.winner;
      winnerProbDisplay.textContent = `${result.winning_probability_pct}%`;

      battingTeamNameDisplay.textContent = `${result.batting_team} (${result.batting_team_abbr || 'Batting'})`;
      battingTeamProbDisplay.textContent = `${result.batting_win_percentage}%`;

      bowlingTeamNameDisplay.textContent = `${result.bowling_team} (${result.bowling_team_abbr || 'Bowling'})`;
      bowlingTeamProbDisplay.textContent = `${result.bowling_win_percentage}%`;

      // Update progress bars with animation
      battingProgressBar.style.width = '0%';
      bowlingProgressBar.style.width = '0%';

      resultContainer.classList.remove('d-none');

      // Small delay for CSS width animation to trigger cleanly
      setTimeout(() => {
        battingProgressBar.style.width = `${result.batting_win_percentage}%`;
        bowlingProgressBar.style.width = `${result.bowling_win_percentage}%`;
      }, 50);

      // Render derived match stats if available
      if (result.stats) {
        statRunsLeft.textContent = result.stats.runs_left;
        statBallsLeft.textContent = result.stats.balls_left;
        statWicketsLeft.textContent = result.stats.wickets_left;
        statCRR.textContent = result.stats.crr;
        statRRR.textContent = result.stats.rrr;
      }

      // Smooth scroll to result section
      resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
      console.error('Fetch error:', err);
      showError('Network error connecting to prediction backend. Make sure the Flask server is running.');
    } finally {
      setLoading(false);
    }
  });
});

