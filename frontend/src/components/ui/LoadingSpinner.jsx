import './LoadingSpinner.css';

/**
 * Modern AI-themed loading spinner with dots animation
 */
export default function LoadingSpinner({ size = 'md', variant = 'dots' }) {
    return (
        <div className={`loading-spinner loading-spinner--${size}`}>
            {variant === 'dots' ? (
                <div className="loading-spinner__dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            ) : (
                <div className="loading-spinner__circle" />
            )}
        </div>
    );
}
