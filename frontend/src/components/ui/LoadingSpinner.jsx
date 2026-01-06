import './LoadingSpinner.css';

/**
 * Loading spinner with accessibility support
 */
export default function LoadingSpinner({ size = 'md', className = '' }) {
    const sizeClasses = {
        sm: 'spinner--sm',
        md: 'spinner--md',
        lg: 'spinner--lg',
    };

    return (
        <div
            className={`spinner ${sizeClasses[size]} ${className}`}
            role="status"
            aria-label="Đang tải..."
        >
            <div className="spinner__circle" />
            <span className="sr-only">Đang tải...</span>
        </div>
    );
}
