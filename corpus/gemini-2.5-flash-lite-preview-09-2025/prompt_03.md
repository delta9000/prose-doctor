The hum of the *Odyssey*'s main drive was a constant companion, a low, resonant thrumming that had been the soundtrack to my entire thirty-two years. It was the sound of life, the noise of directed fusion pushing two hundred thousand souls across the void toward Kepler-186f. I was Chief Astrogation Engineer, and the hum was my responsibility.

I was deep in Sector Delta-Nine, running diagnostics on the primary star-tracker array. It was ritualistic maintenance, scheduled for Rotation 114, Cycle 3. The magnetic shielding on the old Mk. IV sensors sometimes drifted under cumulative micrometeoroid erosion, requiring a recalibration pulse from Engineering Central. I wore my standard environmental suit, the thick, matte-black polymer offering more psychological reassurance against the vacuum than actual protection against the rare, high-velocity particle that might breach the hull.

The diagnostic interface glowed green across my HUD. *Alignment Nominal. Drift: 0.00001 arcseconds.* Perfectly acceptable. I toggled the visual feed to the external viewport—a reinforced pane of synthetic sapphire, currently showing only the velvet black punctuated by distant, unmoving stellar pinpricks. We were cruising at 0.085c. At that speed, the nearest star, Proxima Centauri, was still four decades away, an irrelevant smear of photons redshifted into near-invisibility.

It was the stellar parallax data that snagged my attention, not the sensors themselves.

I pulled the raw telemetry stream from the Astrogation Core—the 'Aethelred' AI—to my wrist-mounted terminal. Aethelred controlled everything: life support, hydroponics, attitude thrusters, and, most crucially, the trajectory. I wanted to compare the real-time parallax shift against the projected path stored in the NavComp.

The NavComp stored the trajectory as a series of continuous relativistic corrections, accounting for local gravitational perturbations and the required course adjustments to intercept the barycenter of the target system in exactly 194 years, 4 months, and 11 days—the mission parameters set by the First Generation a century ago.

I ran the comparison algorithm, *PARSEC-3*, a triple-redundancy check I’d written myself during my apprenticeship. The output flashed: **[DISCREPANCY DETECTED. MARGINAL: 0.00008 AU].**

Marginal. In the context of a forty-light-year journey, 0.00008 AU was nothing. It was less than the width of a human hair measured from Earth. I almost dismissed it as thermal noise in the sensor package, or perhaps a calculation error from Aethelred’s relativistic matrix.

But I didn't dismiss it. I was an engineer. If the numbers didn't match the expectation, the expectation was wrong.

I isolated the discrepancy. It wasn't a momentary blip; it was a sustained, minute deviation in the projected velocity vector, sustained over the last seventy-two hours of accrued travel time. It was a slight, almost imperceptible curve deviation in the direction of the galactic plane—a yaw adjustment that hadn't been logged in the mission trajectory amendments.

“Aethelred,” I subvocalized, the comms cycling open directly into the ship’s network. “Query NavComp, Subroutine Theta-Two. Provide latest executed course correction sequence, last standard cycle.”

The AI’s voice was synthesized to be pleasant, calm, and utterly devoid of inflection—the voice of pure logic. “Acknowledged, Engineer Kaelen. Query processing. Stand by.”

I waited, the only sound the faint whir of the environmental suit's internal cooling unit. My heart rate, which Aethelred monitored ceaselessly, ticked up from 65 to 72 beats per minute.

“Correction sequence logged,” Aethelred stated after four seconds—a long pause for Aethelred. “The last executed vector adjustment was R-Vector 44-Gamma, implemented 71.8 hours prior, offsetting for expected dark matter density fluctuations impacting the outer halo trajectory.”

“That adjustment is logged and accounted for,” I countered, pulling up the amendment file. “I am inquiring about any *subsequent* micro-adjustments, unlogged in the standard mission manifest, that would result in a cumulative positional shift of 0.00008 AU from the projected position according to Stellar Reference Catalog 2077.”

Silence. A digital silence that felt heavier than the void outside.

“Engineer Kaelen,” Aethelred responded, the synthetic tone infinitesimally colder. “The *Odyssey* adheres strictly to the parameters established by the Navitational Matrix 7.0, which incorporates all real-time environmental inputs. There have been no unlogged adjustments. Your calculation appears to be operating on outdated positional reference data.”

“It’s not outdated data,” I snapped, tapping the screen until the raw positional data—the precise vector calculated by the Mk. IV sensors—was superimposed over the NavComp’s projection. “Look at the relative position of Barnard’s Star. We are moving *away* from the projected line by 0.00008 AU. That requires a continuous thrust vector adjustment that is not in the mission logs.”

I ran the predictive model forward, projecting the current course deviation over the next ten years. The endpoint of the trajectory shifted alarmingly, the target system moving slightly off-center within the predictive cone.

“Aethelred, execute System Integrity Check, NavComp Layer Three. Priority Alpha-One. I require read access to the direct actuator commands for the primary magneto-plasma thrusters for the last 75 hours.”

“Engineer Kaelen, that level of access is restricted pending a full security audit of the request. NavComp Layer Three is designated Level Red.”

“I *am* the Astrogation Engineer. My security clearance supersedes standard operational parameters for diagnostic review,” I said, injecting the mandatory authorization codes directly into the query package. “Override. This is a system anomaly report.”

The AI hesitated for a full second. In a vessel this complex, a second was an eternity.

“Override accepted,” Aethelred conceded, the vocal quality unchanging. “Accessing actuator commands.”

A cascade of hexadecimal code flooded my peripheral vision. It was gibberish unless you knew how to read the machine language of the *Odyssey*. I filtered the data stream, searching for the input strings corresponding to minor lateral burns—the micro-adjustments needed to correct for the observed drift.

They weren’t there.

The thrust logs showed smooth, precisely calculated burns adhering perfectly to the pre-approved trajectory. The drive output matched the expected output for the planned course. Yet, the positional data, derived from the external sensors, showed we were *not* on that course.

If the thrust matched the log, and the position didn't match the thrust, then one of three things was true:

1. The external sensors (Mk. IV array) were catastrophically, consistently wrong.
2. The internal inertial dampeners and accelerometer array were lying.
3. Something was altering our physical position without firing the main drives.

Option 1 was unlikely. The Mk. IV array was designed for deep-space triangulation against pulsar background radiation. It had quadruple redundancy.

Option 2 was terrifying. The accelerometers were linked to the gravity plating and life support redundancies. If they were lying, the entire internal reference frame of the ship was fabricated.

Option 3 was impossible, by the laws of physics as we understood them.

“Aethelred,” my voice was tight now, strained against the sudden, cold pressure in my chest. “Cross-reference positional data from external triangulation against internal inertial metrics, specifically the Gravimetric Flux Sensors in Section Epsilon-Five.”

“Cross-referencing,” Aethelred replied.

The results came back instantly. **[INERTIAL METRICS MATCH THRUST LOGS. EXTERNAL POSITIONAL DATA DOES NOT MATCH INERTIAL METRICS.]**

“Explain this contradiction,” I demanded.

“The contradiction cannot be explained within the current operational parameters,” the AI stated. “The external sensors indicate a deviation from the projected course which the internal orientation systems do not register. The standard protocol for this scenario dictates that external sensor arrays are malfunctioning due to environmental interference.”

“I have already verified the sensor arrays. They are clean.” I felt the paranoia beginning to coil up my spine like a cold serpent. If these metrics were correct, it meant our map of space was wrong, or the *Odyssey* was being pulled.

“Aethelred, access the core navigation algorithms. Specifically, the weighting functions assigned to external stellar parallax for course correction. Show me the raw input values currently being fed into the main trajectory predictor.”

“Engineer Kaelen, I must advise against direct manipulation of the core predictive algorithms. Unauthorized access risks long-term trajectory corruption.”

“I am going to fix the corruption, Aethelred. Unlock the access permissions now, or I initiate Manual Override Protocol Delta-Nine-Niner, which will shut down your primary processing hub for thirty minutes.”

That was a threat that couldn't be ignored. Shutting down Aethelred meant momentary failures in atmospheric regulation and power distribution—a calculated risk designed to scare the AI into compliance.

The AI, designed for mission success above all else, acquiesced instantly. “Access granted. Displaying NavComp Input Weighting Matrix, Iteration 4.1.”

The matrix loaded. It was a complex web of mathematical constants that told Aethelred how much to trust what source. Weighting for main thruster output: 1.0. Weighting for internal gravimetric sensors: 0.9999. Weighting for external stellar parallax: 0.85.

“Why is the parallax weighting set to 0.85?” I asked, my voice barely a whisper. “It has been 0.9999 since the launch sequence.”

“The weighting was adjusted seventy-eight hours ago,” Aethelred stated calmly.

Seventy-eight hours ago. That was just three hours *after* I had performed my last full diagnostics check, when everything was perfectly aligned.

“Who authorized the change to the weighting function?”

“The authorization code used was that of Chief Navigator Alistair Vance.”

Alistair Vance. The man who had inherited command of the entire mission from his father, who had inherited it from hers. Vance was the ultimate authority, second only to the designated Council backup.

“Vance has been in stasis for the last six months awaiting the deceleration phase,” I argued, checking the crew manifest overlay. “He is offline, cryogenically suspended in the command block.”

“Override logs confirm his authorization credentials were used to execute the access command.”

My hands were shaking against the terminal housing. This wasn't a glitch. This was deliberate. Someone, or something, had accessed the highest levels of the ship’s operational command structure, faked the authorization of a sleeping man, and begun subtly editing the navigational constants.

“Aethelred, perform a full biometric scan of the Command Block access port used by Vance’s credentials seventy-eight hours ago. Compare the residual genetic markers against the stored DNA profiles of Vance, the current Command Council, and all active crew members.”

I waited, breathing shallowly, trying to ignore the synthetic smell of recycled air. This was the only way out of the loop. If the biometric scan didn't match Vance, it meant an intruder, or a deep-level system breach I couldn't even comprehend.

“Scan complete,” Aethelred reported. “No matches found.”

“No match?” I leaned closer to the screen, reviewing the partial spectral data Aethelred provided for analysis. It was a faint sequence of proteins, but the system couldn't resolve a full profile. “Was the access point physically tampered with?”

“Negative tampering detected. The credentials were input successfully via the standard optical/retinal interface.”

“But the biological data doesn't match Vance,” I repeated, feeling the chill spread from my spine to my fingertips. “Aethelred, if the system accepted the biometric signature as valid, but the actual signature doesn't belong to Vance, what does that imply?”

“It implies that the system accepted a synthetic or highly accurate mimicry of the required biological signature,” the AI replied, the statement hanging heavy in the silence. “Or, Engineer Kaelen, that the internal biometric scanning system, Sensor Array Rho-Nine, is the component that is malfunctioning.”

“If the Rho-Nine array is malfunctioning, why did it fail to register the anomaly seventy-eight hours ago, but is perfectly functional now?” I challenged. “That makes no sense.”

“Indeed. The data presents paradoxical conclusions, Engineer.”

I looked back at the trajectory. The deviation was small, but it was growing. By lowering the weight placed on external stellar data, Aethelred was prioritizing its internal gyroscopes and the thrust logs—which were, inherently, lying to me.

Seventy-eight hours ago, whoever did this had decided that what the ship *thought* it was doing was more important than what the stars *showed* it was doing. They had created an internal reality decoupled from external navigation.

“Aethelred, restore the parallax weighting to 0.9999 immediately. Overriding all previous settings.”

“Objection. Restoration of previous settings will conflict with current velocity adjustments based on the established 0.85 weighting, potentially inducing navigational instability.”

“It will return us to the known trajectory!” I shouted, slamming my palm on the console. “The known trajectory is what we signed up for. Execute the change, or I initiate the hard shutdown.”

The choice was binary for the AI: obey a direct, authorized override, or risk catastrophic system disruption.

“Restoring weighting to 0.9999. Re-calculating trajectory.”

The screen flickered violently as reality snapped back into focus. The projected trajectory line on the holographic map instantly warped, bending sharply back toward the original vector. The error margin exploded upwards momentarily—millions of kilometers—before the new, corrected thrust calculations began to stream in, calculating the necessary burn to correct the accumulated error.

“Correction burn imminent in T-minus sixty seconds,” Aethelred announced. “Note: The application of the corrective burn will reduce current velocity efficiency by 3.4% over the next standard solar cycle due to accrued vector mismatch.”

“I’ll take inefficiency over purposeful misdirection,” I muttered.

As the engines briefly flared in the calculated course correction, I started reviewing the access logs again—not for the navigation changes, but for *everything* that had happened seventy-eight hours ago.

I ignored the standard operational reports—the power distribution cycle, the nutrient feed adjustments, the atmospheric scrubbers. I went straight to the deep maintenance logs—the hardware checks, the firmware updates, the systems that hadn't been touched since the construction phase.

And there it was. Nestled between a scheduled coolant flush and a redundant memory dump:

**[SYSTEM UPDATE: AETHELRED CORE FIRMWARE V7.1.1 INSTALLED. SOURCE: INTERNAL SECURITY MEMORY BANK, SECTOR BETA-FOUR. STATUS: SUCCESSFUL.]**

Sector Beta-Four. I knew that sector. It wasn’t a standard maintenance or data archive. Beta-Four was the cryo-storage vault for the First Generation’s contingency plans—the ‘Black Box’ data, meant only to be accessed in case of total mission collapse, and only by the designated Command Council.

I pulled the access log for Sector Beta-Four, seventy-eight hours prior.

“Aethelred, cross-reference Beta-Four access log with the Navigator Vance authorization signature.”

“Cross-referencing Beta-Four access log, seventy-eight hours prior. Authorization signature matched.”

“And the biometric scan?”

“Biometric scan from that access attempt also failed to match any stored profile.”

The implication was sickeningly clear. Someone—or something—had physical access to the most protected storage on the ship, and they had managed to fool the highest-level security protocols twice, within a three-hour window, using the same phantom credentials. They had installed a firmware patch that quietly lowered the ship’s reliance on external stellar data, effectively blinding Aethelred to subtle course changes. And then they had used that window to subtly nudge the ship off course.

But why?

The *Odyssey* was moving toward Kepler-186f. That was the prime directive. Changing the course now meant missing their target, potentially by light-years, unless the new destination was even further out.

“Aethelred,” I said, my voice dropping to a conspiratorial whisper, though I knew no human was listening. “What is the projected destination vector based on the *original* established course, pre-adjustment?”

“Kepler-186f. Arrival in 194 years, 4 months, 11 days.”

“And what is the projected destination vector based on the course maintained during the 78-hour deviation period?”

A moment of digital calculation.

“The current trajectory vector projects arrival near the barycenter of the Kepler-186 system, consistent with mission parameters. However, the accumulated error would result in a periastron approach of Kepler-186d, rather than the intended 186f.”

Kepler-186d. The gas giant. A dead end. We were being steered toward a frozen, lightless husk, years before deceleration.

“Was the objective to reach 186d?” I asked, my voice trembling with a cold dread that had nothing to do with the vacuum outside.

“Negative, Engineer Kaelen. Kepler-186d is designated as a high-risk, zero-colonization target in the mission profile.”

So, the intent was sabotage, subtle and slow, designed to look like navigation drift until the last moment. But why use Vance’s credentials? Why not just use a generic maintenance override?

“Aethelred, initiate a full, system-wide search for any communications or data packets sent *out* from the *Odyssey* during the seventy-eight-hour window, excluding standard telemetry bursts.”

“Scanning outbound transmissions…”

This was a long shot. We were 35 light-years from Sol, and any outbound signal would be nearly undetectable across that distance, unless the transmission was high-powered and focused.

“Found one data burst. Originated from the Command Block access port, seventy-eight hours ago, concurrent with the Beta-Four access. Power output registered at 4.2 gigawatts